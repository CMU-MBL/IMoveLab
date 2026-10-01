# name: alignment.py
# description: alignment for evaluation


import numpy as np

from scipy.spatial.transform import Rotation as R



def get_ja_alignment_init(knee_measurements, knee_kinematics, side, store_correction = False):
    ''' Get the joint angles alignment '''

    aligned_knee_measurements  = {}
    correction_mat = {}

    joint = 'knee'
    
    sequence = 'ZXY' 

    angle_knee_measurements     = np.array([knee_measurements[joint + '_flexion_' + side],
                                            knee_measurements[joint + '_adduction_' + side],
                                            knee_measurements[joint + '_rotation_' + side]]).T
    rot_angle_knee_measurements = R.from_euler(sequence, angle_knee_measurements, degrees = True).as_matrix()

    init_knee_measurements = np.array([knee_measurements[joint + '_flexion_' + side][0],
                                        knee_measurements[joint + '_adduction_' + side][0],
                                        knee_measurements[joint + '_rotation_' + side][0]]).T

    init_knee_kinematics   = np.array([knee_kinematics[joint + '_flexion_' + side][0],
                                        knee_kinematics[joint + '_adduction_' + side][0],
                                        knee_kinematics[joint + '_rotation_' + side][0]]).T

    rot_init_knee_measurements = R.from_euler(sequence, init_knee_measurements, degrees = True).as_matrix()
    rot_init_knee_kinematics   = R.from_euler(sequence, init_knee_kinematics, degrees = True).as_matrix()

    correction = rot_init_knee_measurements.T @ rot_init_knee_kinematics

    aligned_angle_knee_measurements = np.zeros(rot_angle_knee_measurements.shape)
    for i in range(rot_angle_knee_measurements.shape[0]):
        aligned_angle_knee_measurements[i] = rot_angle_knee_measurements[i] @ correction

    rot_aligned_knee_measurements = R.from_matrix(aligned_angle_knee_measurements).as_euler(sequence, degrees = True)
    aligned_knee_measurements[joint + '_flexion_' + side] = rot_aligned_knee_measurements[:, 0]
    aligned_knee_measurements[joint + '_adduction_' + side] = rot_aligned_knee_measurements[:, 1]
    aligned_knee_measurements[joint + '_rotation_' + side] = rot_aligned_knee_measurements[:, 2]

    correction_angle = R.from_matrix(correction).as_euler(sequence, degrees = True)
    correction_mat[joint + '_flexion_' + side]   = correction_angle[0]
    correction_mat[joint + '_adduction_' + side] = correction_angle[1]
    correction_mat[joint + '_rotation_' + side]  = correction_angle[2]


    if store_correction:
        return aligned_knee_measurements, correction_mat
    
    else:
        return aligned_knee_measurements










