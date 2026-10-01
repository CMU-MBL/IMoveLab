# name: constant_mt.py
# description: constants for Xsens MTw Awinda IMUs


# --- Physical constants --- #
EARTH_G_ACC = 9.81 # m/s^2


# --- Experimental setup --- #
# Sampling rate
MT_SAMPLING_RATE = 40 # Hz (experiment 2)


# --- MTw processing --- #
# Filter list
MT_FILTER_LIST = {'9D': ['Xsens', 'VQF', 'MAD', 'MAH', 'EKF'], 
                  '6D': ['VQF', 'MAD', 'MAH', 'EKF', 'RIANN']}

# Calibration tasks
MT_CALIBRATION_TASKS = ['static', 'treadmill_walking', 'cmj']


# Sensor id
LAB_IMU_NAME_MAP = {'PELVIS':    '00B4D7D3', 
                    'THIGH_L_M': '00B4D7FD', 
                    'SHANK_L_M': '00B4D7CE', 
                    'FOOT_L':    '00B4D7FF', 
                    'THIGH_R_M': '00B4D6D1', 
                    'SHANK_R_M': '00B4D7FB', 
                    'FOOT_R':    '00B4D7FE',
                    'THIGH_R_H': '00B4D7D0', 
                    'THIGH_R_L': '00B4D7D8', 
                    'SHANK_R_H': '00B4D7BA', 
                    'SHANK_R_L': '00B4D7D5', 
                    'THIGH_L_H': '00B4D7D2', 
                    'THIGH_L_L': '00B4D7CD', 
                    'SHANK_L_H': '00B4D7CF', 
                    'SHANK_L_L': '00B4D7FA'}


# For OpenSense
MT_TO_OPENSENSE_MAP = {'pelvis': 'pelvis_imu',
                       'foot_r': 'calcn_r_imu', 'shank_r': 'tibia_r_imu', 'thigh_r': 'femur_r_imu',
                       'foot_l': 'calcn_l_imu', 'shank_l': 'tibia_l_imu', 'thigh_l': 'femur_l_imu'}


# --- Long trial period --- #
# Trial indices
LONG_TRIAL_ID = {'4l': {'t1': {'trial_start': 0,       'trial_end':  95200,  'sitting_start': 63000,  'sitting_end': 94300},
                        't2': {'trial_start': 183000,  'trial_end':  279400, 'sitting_start': 247800, 'sitting_end': 278200},
                        't3': {'trial_start': 369000,  'trial_end':  462200, 'sitting_start': 431400, 'sitting_end': 460800}},
                 '5l': {'t1': {'trial_start': 0,       'trial_end':  95000,  'sitting_start': 64300,  'sitting_end': 93000},
                        't2': {'trial_start': 181000,  'trial_end':  273200, 'sitting_start': 248500, 'sitting_end': 271500},
                        't3': {'trial_start': 357000,  'trial_end':  452500, 'sitting_start': 423000, 'sitting_end': 451000}},
                 '6l': {'t1': {'trial_start': 0,       'trial_end':  100000, 'sitting_start': 62500,  'sitting_end': 97800},
                        't2': {'trial_start': 197000,  'trial_end':  295000, 'sitting_start': 260000, 'sitting_end': 291600},
                        't3': {'trial_start': 386000,  'trial_end':  -1,     'sitting_start': 449250, 'sitting_end': 478000}},
                 '13l': {'t1': {'trial_start': 0,      'trial_end':  93500,  'sitting_start': 62400,  'sitting_end': 91750},
                         't2': {'trial_start': 174000, 'trial_end':  269300, 'sitting_start': 242000, 'sitting_end': 267500},
                         't3': {'trial_start': 355500, 'trial_end':  -1,     'sitting_start': 419000, 'sitting_end': 452000}},
                 '23l': {'t1': {'trial_start': 0,      'trial_end':  98000,  'sitting_start': 64000,  'sitting_end': 94200},
                         't2': {'trial_start': 187400, 'trial_end':  279500, 'sitting_start': 251800, 'sitting_end': 278600},
                         't3': {'trial_start': 371600, 'trial_end':  -1,     'sitting_start': 435400, 'sitting_end': -1}}}


REMOVAL_OF_BAD_MOCAP = {'4l': {'t1': [[59163, 59167], [36560, 36600]],
                               't2': [],
                               't3': [[51560, 51610], [58185, 58200]]},
                        '5l': {'t1': [[28660, 28700], [12175, 12250], [38130, 38150], [58170, 58200], [36290, 36320], [24625, 24680], [62510, 63400], [59550, 59625], [16340, 16380], [21340, 21380], [51790, 51830], [52310, 52340]], 
                               't2': [[60800, 61100], [26720, 26750], [23940, 23980]], 
                               't3': []}, 
                        '6l': {'t1': [[61250, 61550], [17620, 17660], [29460, 29500], [9730, 9760]], 
                               't2': [[58160, 58220], [56375, 56475], [17680, 17720], [19640, 19700], [19310, 19340], [5770, 5790], [29520, 29560], [33520, 33560], [35590, 35630], [47820, 47880], [52100, 52150], [59350, 59425],[60620, 60660], [50100, 50175]], 
                               't3': [[28750, 28850], [40450, 40600], [8800, 9000], [4200, 5400], [45875, 45975], [56400, 56450], [57340, 57380], [56560, 57620], [60200, 60400], [61400, 61600], [46800, 47100], [9940, 9980]]}, 
                        '13l': {'t1': [[35325, 35360]], 
                                't2': [[12900, 12950], [21840, 21900], [37660, 37720], [61285, 61305], [21840, 21900], [57000, 57150], [29480, 29580], [32875, 32950], [34680, 34740]], 
                                't3': [[30400, 30550], [34225, 34325], [20920, 20960], [21180, 21240], [56950, 57050], [5520, 5560], [23660, 23720]]}, 
                        '23l': {'t1': [[56960, 56990], [29945, 29970]], 
                                't2': [], 
                                't3': [[660, 700], [40290, 40320]]}}






