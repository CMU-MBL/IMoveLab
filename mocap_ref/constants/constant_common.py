# name: constant_common.py
# description: common constants used in processing


# --- Directory --- #
IN_LAB_PATH        = 'data/'
IN_MOCAP_PATH      = 'mocap_data/'
IN_MT_PATH         = 'imu_data/'

OPENSENSE_ASSET_PATH = 'opensense_assets/'


# --- Output --- #
OUT_SYNC_INFO           = 'outputs/sync_info/'
OUT_EXERCISE_INDEX_PATH = 'imu_benchmark/outputs/exercise_index/'


# --- File format --- #
MT_EXTENSION    = '.txt'
MOCAP_EXTENSION = '.csv'


# --- Experiment parameters --- #
STATIC_STANDING_PERIOD = 3 # s

NUM_EXERCISE_REPS = 3 # number of repetitions for each (non-locomotion) exercise


# -- Data processing --- #
SUBJECT_LIST_TUNING = [2, 3]
SUBJECT_LIST        = [4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25]


# --- Name mapping --- #
# Task
LAB_TASK_NAME_MAP = {'static':            't0_static_pose_001',
                     'walking':           't1_walking_001',
                     'treadmill_walking': 't2_treadmill_walking_001',
                     'treadmill_running': 't3_treadmill_running_001',
                     'lat_step':          't4_lat_step_001',
                     'step_up_down':      't5_step_up_down_001',
                     'drop_jump':         't6_drop_jump_001',
                     'cmj':               't7_cmjdl_001',
                     'squat':             't8_squat_001',
                     'step_n_hold':       't9_step_n_hold_001',
                     'sls':               't10_sls_001',
                     'sts':               't11_sts_001', 
                     'long_walk1':        't12_longwalk_001',
                     'long_walk2':        't12_longwalk_002',
                     'long_walk3':        't12_longwalk_003'}

MAPPING_TASK_TO_ID = {'static':            0,
                      'walking':           1,
                      'treadmill_walking': 2,
                      'treadmill_running': 3,
                      'lat_step':          4,
                      'step_up_down':      5,
                      'drop_jump':         6,
                      'cmj':               7,
                      'squat':             8,
                      'step_n_hold':       9,
                      'sls':               10,
                      'sts':               11}

MAPPING_TASK_TO_ID_3 = {'static':            0,
                        'treadmill_walking': 2,
                        'treadmill_running': 3,
                        'sts':               11}

LIST_LOCOMOTION_TASK = ['walking', 'treadmill_walking', 'treadmill_running', 'long_walk1', 'long_walk2', 'long_walk3']

# --- Kinematics signs --- #
JA_SIGN = {'pelvis_tilt': 1,        'pelvis_list': 1,       'pelvis_rotation': 1,
           'hip_adduction_l': -1,   'hip_rotation_l': -1,   'hip_flexion_l': 1,
           'knee_adduction_l': -1,  'knee_rotation_l': -1,  'knee_flexion_l': -1, 
           'ankle_adduction_l': -1, 'ankle_rotation_l': -1, 'ankle_flexion_l': 1, 
           'hip_adduction_r': 1,    'hip_rotation_r': 1,    'hip_flexion_r': 1,
           'knee_adduction_r': 1,   'knee_rotation_r': 1,   'knee_flexion_r': -1,
           'ankle_adduction_r': 1,  'ankle_rotation_r': 1,  'ankle_flexion_r': 1}


# --- Tuning --- #
TUNING_SUBJECT_LIST = [2, 3]


# --- Data processing --- #
VERY_HIGH_NUMBER = 999 # to highlight the marker gaps in the processing


# --- Alignment of IMUs and mocap --- #
ALIGNMENT_PERIOD = [0, 100] # first 100 frames of the static task for alignment
ISOLATED_CASES   = {5:  {'treadmill_walking': [150, 200]}, 
                    11: {'treadmill_running': [420, 421]}, 
                    18: {'lat_step': [210, 220], 'walking': [550, 570]}, 
                    7:  {'treadmill_walking': [2000, 2100], 'treadmill_running': [2000, 2100]}} # special cases due to bad mocap 


# --- Joint constraints --- #
HIP_FLEX_LIM    = 180 # relax flexion constraint
HIP_EXT_LIM     = -180 # relax extension constraint
HIP_ADD_LIM     = 30
HIP_ABD_LIM     = -50
HIP_INT_ROT_LIM = 40
HIP_EXT_ROT_LIM = -40

ANKLE_FLEX_LIM    = 180 # relax flexion constraint
ANKLE_EXT_LIM     = -180 # relax extension constraint
ANKLE_ADD_LIM     = 20
ANKLE_ABD_LIM     = -20
ANKLE_INT_ROT_LIM = 20
ANKLE_EXT_ROT_LIM = -20


# --- Constraint gains for soft constraint feedback method --- #
ALPHA_KNEE  = 0.9
KAPPA_HIP   = 0.08**2
KAPPA_ANKLE = 0.5**2






