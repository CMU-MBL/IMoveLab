# name: constant_common.py
# description: common constants for data processing and saving


# --- Data path --- #
DATA_PATH    = 'data/'
IMU_PATH     = 'IMUs/' # raw IMU data from the MC10 Biostamp system
MOCAP_PATH   = 'Vicon/' # raw mocap data from Vicon
BIPLANE_PATH = 'Kinematics/' # exported "kinematics" from the biplane data

HA_DATASET_PATH    = 'HAKnee/' # raw data from the HAKnee dataset

# --- Output path --- #
OUT_MOCAP_IK_PATH = 'outputs/ik/mocap/'
OUT_IMU_IK_PATH   = 'outputs/ik/imu/'

# --- Subject list --- #
HA_SUBJECT_LIST_TUNING = [1, 2] # tuning participants
HA_SUBJECT_LIST_EVAL   = [3, 4, 5, 6, 9, 10, 12, 13, 14, 15, 16, 17, 18] # eval participants

# --- Task list --- #
HA_TASK_MAPPING = {'static': 'static',      # T-pose
                   'shop':   'SHop',        # single-leg hop with 180 degrees rotation
                   'sdrop':  'SDrop',       # single-leg drop jump
                   'ddrop':  'DDrop',       # double-leg drop jump
                   'run':    'runStance',   # run
                   }


HA_NUM_TRIALS = {'static': 1,
                 'ddrop':  3,
                 'sdrop':  3,
                 'shop':   4,
                 'run':    3}


# --- File format --- #
MOCAP_EXTENSION = '.c3d' # extension for mocap data (labelled from Vicon)
IMU_EXTENSION   = '.csv' # extension for IMU data
TT_EXTENSION    = '.xlsx' # extension for trigger times


# --- Body side --- #
BODY_RIGHT = 'R'
BODY_LEFT  = 'L'

# --- IK sign --- #
IK_SIGN = {'hip_adduction_l': -1,   'hip_rotation_l': -1,   'hip_flexion_l': 1,
           'knee_adduction_l': -1,  'knee_rotation_l': -1,  'knee_flexion_l': -1, 
           'ankle_adduction_l': -1, 'ankle_rotation_l': -1, 'ankle_flexion_l': 1, 
           'hip_adduction_r': 1,    'hip_rotation_r': 1,    'hip_flexion_r': 1,
           'knee_adduction_r': 1,   'knee_rotation_r': 1,   'knee_flexion_r': -1,
           'ankle_adduction_r': 1,  'ankle_rotation_r': 1,  'ankle_flexion_r': 1}



STATIC_STANDING_PERIOD = 1


# OpenSense
OPENSENSE_ASSET_PATH  = 'opensense_assets/'
OUT_OPENSENSE_JA_PATH = 'outputs/joint_angles/opensense/'
OUT_OPENSENSE_PATH    = 'outputs/os_ik/'



