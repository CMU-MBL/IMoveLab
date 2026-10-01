# name: event_mocap.py
# description: get gait or repetitive events from mocap data


import numpy as np
from scipy.signal import find_peaks


def get_marker_traj(s_mocap_data):

    ''' Get marker trajectories from mocap data '''

    sacrum_z_r = s_mocap_data['RPS2 Z'].to_numpy()
    sacrum_z_l = s_mocap_data['LPS2 Z'].to_numpy()
    sacrum_z   = (sacrum_z_r + sacrum_z_l)/2

    heel_z_r = s_mocap_data['RCAL Z'].to_numpy()
    heel_x_r = s_mocap_data['RCAL X'].to_numpy()
    heel_z_l = s_mocap_data['LCAL Z'].to_numpy()
    heel_x_l = s_mocap_data['LCAL X'].to_numpy()

    try:
        mt_z_r = s_mocap_data['RMT2 Z'].to_numpy()
        mt_x_r = s_mocap_data['RMT2 X'].to_numpy()
        mt_z_l = s_mocap_data['LMT2 Z'].to_numpy()
        mt_x_l = s_mocap_data['LMT2 X'].to_numpy()
    except:
        mt_z_r = s_mocap_data['R2MT Z'].to_numpy()
        mt_x_r = s_mocap_data['R2MT X'].to_numpy()
        mt_z_l = s_mocap_data['L2MT Z'].to_numpy()
        mt_x_l = s_mocap_data['L2MT X'].to_numpy()

    marker_traj_r   = {'marker_sacrum_z': sacrum_z,
                       'marker_heel_z': heel_z_r,
                       'marker_heel_x': heel_x_r,
                       'marker_mt_z': mt_z_r,
                       'marker_mt_x': mt_x_r}
    marker_traj_l = {'marker_sacrum_z': sacrum_z,
                     'marker_heel_z': heel_z_l,
                     'marker_heel_x': heel_x_l,
                     'marker_mt_z': mt_z_l,
                     'marker_mt_x': mt_x_l}

    return marker_traj_r, marker_traj_l


def ge_heel_toe_sacrum(marker_traj, fs):

    ''' Obtain gait events from the distance between heel and toe/metatarsal markers to the sacrum '''

    min_peak_distance_hc = fs*0.5
    min_peak_distance_to = fs*0.5
    gait_events = {'hc_index': [], 'hc_value': [], 'to_index': [], 'to_value': []}

    heel_marker_z = marker_traj['marker_heel_z']
    toe_marker_z  = marker_traj['marker_mt_z']
    sacrum_marker_z = marker_traj['marker_sacrum_z']

    walking_dir = np.sign(toe_marker_z - heel_marker_z)

    heel_distance_z = heel_marker_z - sacrum_marker_z
    heel_distance_z = walking_dir*heel_distance_z
    temp_hc_index, temp_hc_value = find_peaks(heel_distance_z, height = [0, 1], distance = min_peak_distance_hc)
    hc_index                     = 1*temp_hc_index
    hc_value                     = 1*temp_hc_value['peak_heights']

    toe_distance_z = sacrum_marker_z - toe_marker_z
    toe_distance_z = walking_dir*toe_distance_z
    temp_to_index, temp_to_value = find_peaks(toe_distance_z, height = [0, 1], distance = min_peak_distance_to)
    to_index                     = 1*temp_to_index
    to_value                     = 1*temp_to_value['peak_heights']

    # Remove turns
    dop = np.array([0, 1])

    foot_vec  = np.stack([marker_traj['marker_mt_x'] - marker_traj['marker_heel_x'], marker_traj['marker_mt_z'] - marker_traj['marker_heel_z']], axis = 1)
    length    = np.linalg.norm(foot_vec, axis = 1)
    foot_vec  = foot_vec/length[:, None]
    direction = np.dot(foot_vec, dop)

    hc_index_wo_turn = []
    hc_value_wo_turn = []
    for i in range(len(hc_index) - 1):
        if np.abs(direction[hc_index[i]] - direction[hc_index[i+1]]) < 0.2:
            hc_index_wo_turn.append(hc_index[i])
            hc_value_wo_turn.append(hc_value[i])
    hc_index_wo_turn = np.array(hc_index_wo_turn)
    hc_value_wo_turn = np.array(hc_value_wo_turn)

    # Gait cycle time constraints
    min_gct = 0.6*fs
    max_gct = 1.8*fs

    for i in range(len(hc_index_wo_turn)):
        try:
            temp_id = np.where(to_index > hc_index_wo_turn[i])[0][0]
            if to_index[temp_id] not in gait_events['to_index']:
                if ((to_index[temp_id] - hc_index_wo_turn[i]) > 0.6*min_gct) and ((to_index[temp_id] - hc_index_wo_turn[i]) < 0.6*max_gct):
                    gait_events['hc_index'].append(hc_index_wo_turn[i])
                    gait_events['hc_value'].append(hc_value_wo_turn[i])

                    gait_events['to_index'].append(to_index[temp_id])
                    gait_events['to_value'].append(to_value[temp_id])
        except:
            pass

    # Remove bouts with less than 2 gait cycles at the beginning and end
    start_id = 0
    cycle_count = 0
    for i in range(len(gait_events['hc_index']) - 1):
        if ((gait_events['hc_index'][i+1] - gait_events['hc_index'][i]) > min_gct) and ((gait_events['hc_index'][i+1] - gait_events['hc_index'][i]) < max_gct):
            cycle_count += 1
        else:
            cycle_count = 0
        
        if cycle_count >= 3:
            start_id = i - cycle_count + 1
            break
    
    stop_id = len(gait_events['hc_index']) - 1
    cycle_count = 0
    for i in range(len(gait_events['hc_index']) - 1, 0, -1):
        if ((gait_events['hc_index'][i] - gait_events['hc_index'][i-1]) > min_gct) and ((gait_events['hc_index'][i] - gait_events['hc_index'][i-1]) < max_gct):
            cycle_count += 1
        else:
            cycle_count = 0
        
        if cycle_count >= 3:
            stop_id = i + cycle_count
            break
    

    gait_events['hc_index'] = gait_events['hc_index'][start_id:stop_id]
    gait_events['hc_value'] = gait_events['hc_value'][start_id:stop_id]
    gait_events['to_index'] = gait_events['to_index'][start_id:stop_id]
    gait_events['to_value'] = gait_events['to_value'][start_id:stop_id]

    # Remove heel-contact at the beginning of each cycle
    first_hc_id = [0]
    for i in range(1, len(gait_events['hc_index'])):
        if (gait_events['hc_index'][i] - gait_events['hc_index'][i-1]) > max_gct:
            first_hc_id.append(i)
    
    for i in range(len(first_hc_id) - 1, -1, -1):
        gait_events['hc_index'] = np.delete(gait_events['hc_index'], first_hc_id[i])
        gait_events['hc_value'] = np.delete(gait_events['hc_value'], first_hc_id[i])


    gait_events['hc_index'] = np.array(gait_events['hc_index'])
    gait_events['hc_value'] = np.array(gait_events['hc_value'])
    gait_events['to_index'] = np.array(gait_events['to_index'])
    gait_events['to_value'] = np.array(gait_events['to_value'])


    return gait_events





