# name: sfa_cf.py
# description: apply sensor fusion algorithms with constraint feedback for estimating orientation


import quaternion
import numpy as np
import time

from ahrs.filters import Mahony, Madgwick, EKF
from vqf import PyVQF
from riann.riann import RIANN
from ahrs.common.orientation import acc2q, am2q, ecompass
from scipy.spatial.transform import Rotation as R

from constants import constant_common


# NOTE: no constraint feedback for Xsens or RIANN as they don't provide access to the hidden states for feedback


def init_orientation_ahrs(data_main_mt, num_samples, use_acc=True, use_mag=False, f_type=None):

    ''' get the initial orientation for (AHRS) state-estimation filters
    '''

    orientation = {}

    for sensor_name in data_main_mt.keys():

        orientation[sensor_name] = np.zeros((num_samples, 4))

        if use_mag:
            acc0 = data_main_mt[sensor_name].loc[0, ['Acc_X','Acc_Y','Acc_Z']].to_numpy()
            mag0 = data_main_mt[sensor_name].loc[0, ['Mag_X','Mag_Y','Mag_Z']].to_numpy()

            if f_type == 'MAD':
                orientation[sensor_name][0] = ecompass(acc0, mag0, frame = 'NED', representation = 'quaternion')
            else:
                orientation[sensor_name][0] = am2q(acc0, mag0, frame = 'ENU')

        elif use_acc:
            acc0 = data_main_mt[sensor_name].loc[0, ['Acc_X','Acc_Y','Acc_Z']].to_numpy()
            orientation[sensor_name][0] = acc2q(acc0)

        else:
            orientation[sensor_name][0] = np.array([1, 0, 0, 0])

    return orientation


def one_step_update_ahrs(filter, f_type, data, Q, sensor_name, t, use_mag = False):

    ''' get the next orientation using AHRS state-estimation filters
    '''

    acc_t = data[sensor_name].loc[t, ['Acc_X','Acc_Y','Acc_Z']].to_numpy()
    gyr_t = data[sensor_name].loc[t, ['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()
    mag_t = data[sensor_name].loc[t, ['Mag_X','Mag_Y','Mag_Z']].to_numpy() if use_mag else None

    if f_type in ['MAD', 'MAH']:
        start_time = time.time()
        if use_mag:
            Q_ = filter.updateMARG(Q, gyr = gyr_t, acc = acc_t, mag = mag_t)
        else:
            Q_ = filter.updateIMU(Q, gyr = gyr_t, acc = acc_t)
        time_update = time.time() - start_time

    elif f_type == 'EKF':
        start_time = time.time()
        if use_mag:
            Q_ = filter.update(Q, gyr = gyr_t, acc = acc_t, mag = mag_t)
        else:
            Q_ = filter.update(Q, gyr = gyr_t, acc = acc_t)
        time_update = time.time() - start_time

    return Q_, time_update


def get_sensor_transform(initial_orientation, Q, num_static_samples):

    ''' to align sensors with body segments based on the perfect standing assumption for 6D filters '''

    static_orientation = 1*quaternion.as_quat_array(Q[0:num_static_samples]) 

    return initial_orientation * np.mean(static_orientation).conjugate()


def get_joint_rotation(segment_prox, segment_dist, seg2sens_prox, seg2sens_dist):

    ''' to get the joint rotation (quaternion) from the two segments '''
    
    segment_prox_aligned = segment_prox * quaternion.from_rotation_matrix(seg2sens_prox).conjugate()
    segment_dist_aligned = segment_dist * quaternion.from_rotation_matrix(seg2sens_dist).conjugate()

    joint_quat = segment_prox_aligned.conjugate() * segment_dist_aligned

    return joint_quat


def get_all_joints(seg2sens, sensor_frame, timestep = None):

    ''' to get joint rotations of all joints at a given timestep'''

    joint_frame = {}

    joint_frame['hip_r']   = get_joint_rotation(sensor_frame['pelvis'][timestep], sensor_frame['thigh_r'][timestep], seg2sens['pelvis'], seg2sens['thigh_r'])
    joint_frame['knee_r']  = get_joint_rotation(sensor_frame['thigh_r'][timestep], sensor_frame['shank_r'][timestep], seg2sens['thigh_r'], seg2sens['shank_r'])
    joint_frame['ankle_r'] = get_joint_rotation(sensor_frame['shank_r'][timestep], sensor_frame['foot_r'][timestep], seg2sens['shank_r'], seg2sens['foot_r'])
    joint_frame['hip_l']   = get_joint_rotation(sensor_frame['pelvis'][timestep], sensor_frame['thigh_l'][timestep], seg2sens['pelvis'], seg2sens['thigh_l'])
    joint_frame['knee_l']  = get_joint_rotation(sensor_frame['thigh_l'][timestep], sensor_frame['shank_l'][timestep], seg2sens['thigh_l'], seg2sens['shank_l'])
    joint_frame['ankle_l'] = get_joint_rotation(sensor_frame['shank_l'][timestep], sensor_frame['foot_l'][timestep], seg2sens['shank_l'], seg2sens['foot_l'])

    return joint_frame


def correct_heading(joint_quat, seg2sens, adaptor_aligned, sensor_transforms, timestep, joint, prox, dist, alpha, cons_flag = False):

    ''' correct heading based on feedback from biomechanical constraints '''

    joint_rot = R.from_quat(quaternion.as_float_array(joint_quat[joint]), scalar_first = True)
    euler_flex_ext, euler_add_abd, euler_rot = joint_rot.as_euler('ZXY', degrees = True)

    # enforce constraints (Rajagopal et al. 2016)
    if 'hip' in joint:
        if constant_common.JA_SIGN[f'hip_flexion_{joint[-1]}']*euler_flex_ext > constant_common.HIP_FLEX_LIM:
            alpha_flex_ext  = 1.0
            target_flex_ext = constant_common.JA_SIGN[f'hip_flexion_{joint[-1]}']*constant_common.HIP_FLEX_LIM
        elif constant_common.JA_SIGN[f'hip_flexion_{joint[-1]}']*euler_flex_ext < constant_common.HIP_EXT_LIM:
            alpha_flex_ext  = 1.0
            target_flex_ext = constant_common.JA_SIGN[f'hip_flexion_{joint[-1]}']*constant_common.HIP_EXT_LIM
        else:
            alpha_flex_ext  = 0.0
            target_flex_ext = 1*euler_flex_ext

        if constant_common.JA_SIGN[f'hip_adduction_{joint[-1]}']*euler_add_abd > constant_common.HIP_ADD_LIM:
            alpha_add_abd  = 1.0
            target_add_abd = constant_common.JA_SIGN[f'hip_adduction_{joint[-1]}']*constant_common.HIP_ADD_LIM
        elif constant_common.JA_SIGN[f'hip_adduction_{joint[-1]}']*euler_add_abd < constant_common.HIP_ABD_LIM:
            alpha_add_abd  = 1.0
            target_add_abd = constant_common.JA_SIGN[f'hip_adduction_{joint[-1]}']*constant_common.HIP_ABD_LIM
        else:
            # alpha_add_abd  = 0.0
            # target_add_abd = 1*euler_add_abd
            alpha_add_abd  = 1.0*alpha
            target_add_abd = 0.0
        
        if constant_common.JA_SIGN[f'hip_rotation_{joint[-1]}']*euler_rot > constant_common.HIP_INT_ROT_LIM: 
            alpha_rot  = 1.0
            target_rot = constant_common.JA_SIGN[f'hip_rotation_{joint[-1]}']*constant_common.HIP_INT_ROT_LIM
        elif constant_common.JA_SIGN[f'hip_rotation_{joint[-1]}']*euler_rot < constant_common.HIP_EXT_ROT_LIM:
            alpha_rot  = 1.0
            target_rot = constant_common.JA_SIGN[f'hip_rotation_{joint[-1]}']*constant_common.HIP_EXT_ROT_LIM
        else:
            alpha_rot  = 1.0*alpha
            target_rot = 0.0

    elif 'ankle' in joint:
        if constant_common.JA_SIGN[f'ankle_flexion_{joint[-1]}']*euler_flex_ext > constant_common.ANKLE_FLEX_LIM:
            alpha_flex_ext  = 1.0
            target_flex_ext = constant_common.JA_SIGN[f'ankle_flexion_{joint[-1]}']*constant_common.ANKLE_FLEX_LIM
        elif constant_common.JA_SIGN[f'ankle_flexion_{joint[-1]}']*euler_flex_ext < constant_common.ANKLE_EXT_LIM:
            alpha_flex_ext  = 1.0
            target_flex_ext = constant_common.JA_SIGN[f'ankle_flexion_{joint[-1]}']*constant_common.ANKLE_EXT_LIM
        else:
            alpha_flex_ext  = 0.0
            target_flex_ext = 1*euler_flex_ext

        if constant_common.JA_SIGN[f'ankle_adduction_{joint[-1]}']*euler_add_abd > constant_common.ANKLE_ADD_LIM: 
            alpha_add_abd  = 1.0
            target_add_abd = constant_common.JA_SIGN[f'ankle_adduction_{joint[-1]}']*constant_common.ANKLE_ADD_LIM
        elif constant_common.JA_SIGN[f'ankle_adduction_{joint[-1]}']*euler_add_abd < constant_common.ANKLE_ABD_LIM:
            alpha_add_abd  = 1.0
            target_add_abd = constant_common.JA_SIGN[f'ankle_adduction_{joint[-1]}']*constant_common.ANKLE_ABD_LIM
        else:
            # alpha_add_abd  = 0.0
            # target_add_abd = 1*euler_add_abd
            alpha_add_abd  = 1.0*alpha
            target_add_abd = 0.0

        if constant_common.JA_SIGN[f'ankle_rotation_{joint[-1]}']*euler_rot > constant_common.ANKLE_INT_ROT_LIM: 
            alpha_rot  = 1.0
            target_rot = constant_common.JA_SIGN[f'ankle_rotation_{joint[-1]}']*constant_common.ANKLE_INT_ROT_LIM
        elif constant_common.JA_SIGN[f'ankle_rotation_{joint[-1]}']*euler_rot < constant_common.ANKLE_EXT_ROT_LIM:
            alpha_rot  = 1.0
            target_rot = constant_common.JA_SIGN[f'ankle_rotation_{joint[-1]}']*constant_common.ANKLE_EXT_ROT_LIM
        else:
            alpha_rot  = 1.0*alpha
            target_rot = 0.0

    euler_mismatch       = np.array([alpha_flex_ext*(target_flex_ext - euler_flex_ext), 
                                     alpha_add_abd*(target_add_abd - euler_add_abd), 
                                     alpha_rot*(target_rot - euler_rot)])
    rot_heading_mismatch = R.from_euler('ZXY', euler_mismatch, degrees = True)


    q_heading_mismatch = rot_heading_mismatch.as_quat(scalar_first = True)
    q_heading_mismatch = quaternion.as_quat_array(q_heading_mismatch)

    # apply the heading correction to the adaptor
    corrected_adaptor_joint   = joint_quat[joint] * q_heading_mismatch  
    corrected_adaptor_aligned = adaptor_aligned[prox][timestep] * quaternion.from_rotation_matrix(seg2sens[prox]).conjugate() * corrected_adaptor_joint * quaternion.from_rotation_matrix(seg2sens[dist])
    corrected_adaptor_raw     = quaternion.as_float_array(sensor_transforms[dist].conjugate() * corrected_adaptor_aligned)
    corrected_adaptor_raw     /= np.linalg.norm(corrected_adaptor_raw)

    return corrected_adaptor_aligned, corrected_adaptor_raw


def correct_nonsagittal_knee(joint_quat, seg2sens, joint_aligned, sensor_transforms, timestep, joint, prox, dist, alpha):

    ''' correct knee adduction/abduction and internal/external rotation based on the knee coupling (Walker, Rovick & Robertson, J Biomech 1988;21:965-974, eqs 1-2; fitted to cadaver data of Reuben et al. 1986)'''

    joint_rot = R.from_quat(quaternion.as_float_array(joint_quat[joint]), scalar_first = True)
    knee_flex, knee_add, knee_rot = joint_rot.as_euler('ZXY', degrees = True)

    knee_flex *= -1
    if joint[-1] == 'l': 
        knee_add *= -1 
        knee_rot *= -1

    knee_add_coupled = 0.0791*knee_flex - 5.733e-4*knee_flex**2 - 7.682e-6*knee_flex**3 + 5.759e-8*knee_flex**4
    knee_rot_coupled = 0.3695*knee_flex - 2.958e-3*knee_flex**2 + 7.666e-6*knee_flex**3

    knee_add_error = alpha * (knee_add_coupled - knee_add)
    knee_rot_error = alpha * (knee_rot_coupled - knee_rot)

    if joint[-1] == 'l':
        knee_add_error *= -1 
        knee_rot_error *= -1

    rot_knee_error = R.from_euler('ZXY', np.array([0, knee_add_error, knee_rot_error]), degrees = True)
    q_knee_error = rot_knee_error.as_quat(scalar_first = True)
    q_knee_error /= np.linalg.norm(q_knee_error) 
    q_knee_error = quaternion.as_quat_array(q_knee_error)

    corrected_joint         = joint_quat[joint] * q_knee_error
    corrected_joint_aligned = joint_aligned[prox][timestep] * quaternion.from_rotation_matrix(seg2sens[prox]).conjugate() * corrected_joint * quaternion.from_rotation_matrix(seg2sens[dist])
    corrected_joint_raw     = quaternion.as_float_array(sensor_transforms[dist].conjugate() * corrected_joint_aligned)
    corrected_joint_raw     /= np.linalg.norm(corrected_joint_raw)


    return corrected_joint_aligned, corrected_joint_raw


def inject_correction(vqf_filter, corrected_raw, use_mag = False):

    ''' force a VQF filter to report corrected_raw as its current orientation '''

    state  = vqf_filter.state
    target = corrected_raw/np.linalg.norm(corrected_raw)

    if use_mag:
        target = vqf_filter.quatApplyDelta(target, -state['delta'])

    state['accQuat'] = vqf_filter.quatMultiply(target, vqf_filter.quatConj(state['gyrQuat']))
    vqf_filter.state = state


def apply_vqf_cf(data_main_mt, seg2sens, initial_orientation, dim = '6D', fs = 100, params = None):

    ''' Apply VQF with constraint feedback to get corrected orientation '''

    dim     = dim.upper()
    use_mag = (dim == '9D')

    if use_mag:
        for sensor_name in data_main_mt.keys():
            assert set(['Mag_X', 'Mag_Y', 'Mag_Z']).issubset(data_main_mt[sensor_name].columns), \
                f'9D constraint feedback needs magnetometer data, missing for {sensor_name}'

    num_static_samples = constant_common.STATIC_STANDING_PERIOD*fs
    num_samples        = len(data_main_mt['pelvis']['Acc_X'])

    filter = {}
    for sensor_name in data_main_mt.keys():
        filter[sensor_name] = PyVQF(gyrTs = 1.0/fs, tauAcc = params[0], tauMag = params[1])
        
    sensor_transforms = {}

    filter_raw = {}
    for sensor_name in data_main_mt.keys():
        filter_raw[sensor_name] = np.empty((num_samples, 4))

    filter_aligned    = {}

    # no initialization for VQF, need warm-up for the first few seconds to get good estimates
    print('Warm start for VQF convergence')
    for sensor_name in data_main_mt.keys():
        for i in range(10):
            for phantom_timestep in range(num_static_samples):
                gyr_t = data_main_mt[sensor_name].loc[phantom_timestep, ['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()
                acc_t = data_main_mt[sensor_name].loc[phantom_timestep, ['Acc_X','Acc_Y','Acc_Z']].to_numpy()

                filter[sensor_name].update(gyr = gyr_t, acc = acc_t)
                filter_raw[sensor_name][phantom_timestep] = 1*(filter[sensor_name].getQuat9D() if use_mag
                                                               else filter[sensor_name].getQuat6D())

    timestep = 0

    for sensor_name in filter_raw.keys(): # XXX just added
        filter_aligned[sensor_name] = 1*quaternion.as_quat_array(filter_raw[sensor_name])

    # apply the sensor transform to the whole trial
    for sensor_name in data_main_mt.keys():
        if use_mag:
            sensor_transforms[sensor_name] = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            sensor_transforms[sensor_name] = get_sensor_transform(initial_orientation[sensor_name], filter_raw[sensor_name], num_static_samples)

        filter_aligned[sensor_name] = 1*quaternion.as_quat_array(filter_raw[sensor_name])
        for k in range(num_samples):
            filter_aligned[sensor_name][k] = sensor_transforms[sensor_name] * filter_aligned[sensor_name][k]


    last_check = 1*timestep
    update_interval = 1

    time_mt = {'filtering': {sensor_name: [] for sensor_name in data_main_mt.keys()}, 
               'correction': {joint: [] for joint in ['hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l']}}

    while timestep < num_samples:

        if timestep % 1000 == 0:
            print(f'current timestep: {timestep} / {num_samples}')

        # update the filter_raw and filter_aligned every timestep
        for sensor_name in data_main_mt.keys():
            gyr_t = data_main_mt[sensor_name].loc[timestep, ['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()
            acc_t = data_main_mt[sensor_name].loc[timestep, ['Acc_X','Acc_Y','Acc_Z']].to_numpy()

            if use_mag:
                mag_t = data_main_mt[sensor_name].loc[timestep, ['Mag_X','Mag_Y','Mag_Z']].to_numpy()

                start_time = time.time()
                filter[sensor_name].update(gyr = gyr_t, acc = acc_t, mag = mag_t)
                time_mt['filtering'][sensor_name].append(time.time() - start_time)

                filter_raw[sensor_name][timestep] = 1*filter[sensor_name].getQuat9D()

            else:
                start_time = time.time()
                filter[sensor_name].update(gyr = gyr_t, acc = acc_t)
                time_mt['filtering'][sensor_name].append(time.time() - start_time)

                filter_raw[sensor_name][timestep] = 1*filter[sensor_name].getQuat6D()

            filter_aligned[sensor_name][timestep] = 1*quaternion.as_quat_array(filter_raw[sensor_name][timestep])
            filter_aligned[sensor_name][timestep] = sensor_transforms[sensor_name] * filter_aligned[sensor_name][timestep]


        if timestep - last_check >= update_interval:

            joint_quat = get_all_joints(seg2sens, filter_aligned, timestep = timestep)

            cons_flag = False # NOTE: no conservative constraints

            start_time = time.time()
            corrected_adaptor_aligned, corrected_adaptor_raw = correct_heading(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'hip_r', prox = 'pelvis', dist = 'thigh_r', alpha = constant_common.KAPPA_HIP, cons_flag = cons_flag)
            filter_aligned['thigh_r'][timestep] = 1*corrected_adaptor_aligned
            filter_raw['thigh_r'][timestep]     = 1*corrected_adaptor_raw
            inject_correction(filter['thigh_r'], corrected_adaptor_raw, use_mag)
            time_mt['correction']['hip_r'].append(time.time() - start_time)

            start_time = time.time()
            corrected_joint_aligned, corrected_joint_raw = correct_nonsagittal_knee(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'knee_r', prox = 'thigh_r', dist = 'shank_r', alpha = constant_common.ALPHA_KNEE)
            filter_aligned['shank_r'][timestep] = 1*corrected_joint_aligned
            filter_raw['shank_r'][timestep]     = 1*corrected_joint_raw
            inject_correction(filter['shank_r'], corrected_joint_raw, use_mag)
            time_mt['correction']['knee_r'].append(time.time() - start_time)

            start_time = time.time()
            corrected_joint_aligned, corrected_joint_raw = correct_heading(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'ankle_r', prox = 'shank_r', dist = 'foot_r', alpha = constant_common.KAPPA_ANKLE, cons_flag = cons_flag)
            filter_aligned['foot_r'][timestep] = 1*corrected_joint_aligned
            filter_raw['foot_r'][timestep]     = 1*corrected_joint_raw
            inject_correction(filter['foot_r'], corrected_joint_raw, use_mag)
            time_mt['correction']['ankle_r'].append(time.time() - start_time)

            # left side
            start_time = time.time()
            corrected_adaptor_aligned, corrected_adaptor_raw = correct_heading(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'hip_l', prox = 'pelvis', dist = 'thigh_l', alpha = constant_common.KAPPA_HIP, cons_flag = cons_flag)
            filter_aligned['thigh_l'][timestep] = 1*corrected_adaptor_aligned
            filter_raw['thigh_l'][timestep]     = 1*corrected_adaptor_raw
            inject_correction(filter['thigh_l'], corrected_adaptor_raw, use_mag)
            time_mt['correction']['hip_l'].append(time.time() - start_time)

            start_time = time.time()
            corrected_joint_aligned, corrected_joint_raw = correct_nonsagittal_knee(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'knee_l', prox = 'thigh_l', dist = 'shank_l', alpha = constant_common.ALPHA_KNEE)
            filter_aligned['shank_l'][timestep] = 1*corrected_joint_aligned
            filter_raw['shank_l'][timestep]     = 1*corrected_joint_raw
            inject_correction(filter['shank_l'], corrected_joint_raw, use_mag)
            time_mt['correction']['knee_l'].append(time.time() - start_time)

            start_time = time.time()
            corrected_joint_aligned, corrected_joint_raw = correct_heading(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'ankle_l', prox = 'shank_l', dist = 'foot_l', alpha = constant_common.KAPPA_ANKLE, cons_flag = cons_flag)
            filter_aligned['foot_l'][timestep] = 1*corrected_joint_aligned
            filter_raw['foot_l'][timestep]     = 1*corrected_joint_raw
            inject_correction(filter['foot_l'], corrected_joint_raw, use_mag)
            time_mt['correction']['ankle_l'].append(time.time() - start_time)

            last_check = 1*timestep

        timestep += 1

    return filter_aligned, time_mt 


def apply_ahrs_cf(data_main_mt, f_type, seg2sens, initial_orientation, dim = '6D', fs = 100, params = None):
    ''' Apply AHRS filters with constraint feedback to get corrected orientation
    '''
    
    dim     = dim.upper()
    use_mag = (dim == '9D')

    if use_mag:
        for sensor_name in data_main_mt.keys():
            assert set(['Mag_X', 'Mag_Y', 'Mag_Z']).issubset(data_main_mt[sensor_name].columns), \
                f'9D constraint feedback needs magnetometer data, missing for {sensor_name}'

    num_static_samples = constant_common.STATIC_STANDING_PERIOD*fs
    num_samples        = len(data_main_mt['pelvis']['Acc_X'])

    def build_filter():
        if f_type == 'MAD':
            return Madgwick(frequency = fs, gain = params[0])
        
        elif f_type == 'MAH':
            return Mahony(frequency = fs, kp = params[0], ki = params[1])
        
        elif f_type == 'EKF':
            if use_mag:
                ekf = EKF(frequency = fs, noises = [params[0]**2, params[1]**2, params[2]**2],
                          frame = 'ENU', mag = np.empty((0, 3)))
                ekf.a_ref = -ekf.a_ref

                return ekf
            
            return EKF(frequency = fs, noises = [params[0]**2, params[1]**2, params[2]**2]) # NOTE: no ENU for 6D cause we don't know where is North without magnetometer
        
        raise ValueError(f'constraint feedback is not implemented for filter {f_type}')

    if use_mag:
        filter = {sensor_name: build_filter() for sensor_name in data_main_mt.keys()}
    else:
        shared = build_filter()
        filter = {sensor_name: shared for sensor_name in data_main_mt.keys()}

    sensor_transforms = {}

    filter_raw     = init_orientation_ahrs(data_main_mt, num_samples, use_acc = True, use_mag = use_mag, f_type = f_type)
    filter_aligned = {}

    timestep = 1

    while timestep < num_static_samples:
        for sensor_name in data_main_mt.keys():
            filter_raw[sensor_name][timestep], _ = one_step_update_ahrs(filter[sensor_name], f_type, data_main_mt, filter_raw[sensor_name][timestep-1], sensor_name, timestep, use_mag)

        timestep += 1

    for sensor_name in data_main_mt.keys():
        if use_mag:
            sensor_transforms[sensor_name] = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            sensor_transforms[sensor_name] = get_sensor_transform(initial_orientation[sensor_name], filter_raw[sensor_name], num_static_samples)

        filter_aligned[sensor_name] = 1*quaternion.as_quat_array(filter_raw[sensor_name])
        for k in range(num_samples):
            filter_aligned[sensor_name][k] = sensor_transforms[sensor_name] * filter_aligned[sensor_name][k]

    last_check = 1*timestep
    update_interval = 1

    time_mt = {'filtering': {'pelvis': [], 'thigh_r': [], 'shank_r': [], 'foot_r': [], 'thigh_l': [], 'shank_l': [], 'foot_l': []}, 
               'correction': {'hip_r': [], 'knee_r': [], 'ankle_r': [], 'hip_l': [], 'knee_l': [], 'ankle_l': []}}   

    while timestep < num_samples:

        if timestep % 1000 == 0:
            print(f'current timestep: {timestep} / {num_samples}')

        # update the filter_raw and filter_aligned every timestep
        for sensor_name in data_main_mt.keys():
            
            filter_raw[sensor_name][timestep], time_update = one_step_update_ahrs(filter[sensor_name], f_type, data_main_mt, filter_raw[sensor_name][timestep-1], sensor_name, timestep, use_mag)
            time_mt['filtering'][sensor_name].append(time_update)

            filter_aligned[sensor_name][timestep] = 1*quaternion.as_quat_array(filter_raw[sensor_name][timestep])
            filter_aligned[sensor_name][timestep] = sensor_transforms[sensor_name] * filter_aligned[sensor_name][timestep]


        if timestep - last_check >= update_interval:

            joint_quat = get_all_joints(seg2sens, filter_aligned, timestep = timestep)

            # right side
            start_time = time.time()
            corrected_adaptor_aligned, corrected_adaptor_raw = correct_heading(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'hip_r', prox = 'pelvis', dist = 'thigh_r', alpha = constant_common.KAPPA_HIP)
            filter_aligned['thigh_r'][timestep] = 1*corrected_adaptor_aligned
            filter_raw['thigh_r'][timestep]     = 1*corrected_adaptor_raw 
            time_mt['correction']['hip_r'].append(time.time() - start_time)

            start_time = time.time()
            corrected_joint_aligned, corrected_joint_raw = correct_nonsagittal_knee(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'knee_r', prox = 'thigh_r', dist = 'shank_r', alpha = constant_common.ALPHA_KNEE)
            filter_aligned['shank_r'][timestep] = 1*corrected_joint_aligned
            filter_raw['shank_r'][timestep]     = 1*corrected_joint_raw 
            time_mt['correction']['knee_r'].append(time.time() - start_time)

            start_time = time.time()
            corrected_joint_aligned, corrected_joint_raw = correct_heading(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'ankle_r', prox = 'shank_r', dist = 'foot_r', alpha = constant_common.KAPPA_ANKLE)
            filter_aligned['foot_r'][timestep] = 1*corrected_joint_aligned
            filter_raw['foot_r'][timestep]     = 1*corrected_joint_raw 
            time_mt['correction']['ankle_r'].append(time.time() - start_time)

            # left side
            start_time = time.time()
            corrected_adaptor_aligned, corrected_adaptor_raw = correct_heading(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'hip_l', prox = 'pelvis', dist = 'thigh_l', alpha = constant_common.KAPPA_HIP)
            filter_aligned['thigh_l'][timestep] = 1*corrected_adaptor_aligned
            filter_raw['thigh_l'][timestep]     = 1*corrected_adaptor_raw 
            time_mt['correction']['hip_l'].append(time.time() - start_time)

            start_time = time.time()
            corrected_joint_aligned, corrected_joint_raw = correct_nonsagittal_knee(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'knee_l', prox = 'thigh_l', dist = 'shank_l', alpha = constant_common.ALPHA_KNEE)
            filter_aligned['shank_l'][timestep] = 1*corrected_joint_aligned
            filter_raw['shank_l'][timestep]     = 1*corrected_joint_raw 
            time_mt['correction']['knee_l'].append(time.time() - start_time)

            start_time = time.time()
            corrected_joint_aligned, corrected_joint_raw = correct_heading(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'ankle_l', prox = 'shank_l', dist = 'foot_l', alpha = constant_common.KAPPA_ANKLE)
            filter_aligned['foot_l'][timestep] = 1*corrected_joint_aligned
            filter_raw['foot_l'][timestep]     = 1*corrected_joint_raw
            time_mt['correction']['ankle_l'].append(time.time() - start_time)

            last_check = 1*timestep

        timestep += 1

    return filter_aligned, time_mt 














