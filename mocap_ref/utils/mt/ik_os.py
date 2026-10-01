# name: ik_os.py
# description: Perform (constrained) inverse kinematics for IMU data using OpenSense (OpenSim)


import opensim as osim
import numpy as np
import pandas as pd
import quaternion
import math

from bs4 import BeautifulSoup
from scipy.spatial.transform import Rotation as R

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_mt, constant_common



def convert_imu_orientation_to_os(task, orientation_mt, fs, stat_flag, catch_path):

	''' Convert IMU orientation to the OpenSim format (i.e., .sto) '''

	time_id = np.arange(0, orientation_mt['pelvis'].shape[0], 1)/fs

	output = pd.DataFrame()
	output['time'] = time_id

	for segment in orientation_mt.keys():
		quat = quaternion.as_float_array(orientation_mt[segment])
		output_key = constant_mt.MT_TO_OPENSENSE_MAP[segment]
		output[output_key] = ["{},{},{},{}".format(*row) for row in quat]

	if stat_flag:
		orientation_fn = f'{task}_cal_orientation.sto'
	else:
		orientation_fn = f'{task}_orientation.sto'

	with open(catch_path + orientation_fn, 'w') as f:
		f.write('DataRate={}\n'.format(fs))
		f.write('DataType=Quaternion\n')
		f.write('version=3\n')
		f.write('OpenSimVersion=4.4-2022-07-23-0e9fedc\n')
		f.write('endheader\n')
		output.to_csv(f, sep='\t', index=False)


def os_calibration_default(orientation_cal_fn, os_model, visulizeCalibration):

	''' Perform calibration for the OpenSim model with the given orientation in the .sto file '''

	modelFileName               = os_model + '.osim'
	orientationsFileName        = constant_common.OPENSENSE_ASSET_PATH + orientation_cal_fn
	sensor_to_opensim_rotations = osim.Vec3(-math.pi/2, 0, 0)
	baseIMUName                 = 'pelvis_imu'
	baseIMUHeading              = '-z'

	imuPlacer = osim.IMUPlacer()
	imuPlacer.set_model_file(modelFileName)
	imuPlacer.set_orientation_file_for_calibration(orientationsFileName)
	imuPlacer.set_sensor_to_opensim_rotations(sensor_to_opensim_rotations)
	imuPlacer.set_base_imu_label(baseIMUName)
	imuPlacer.set_base_heading_axis(baseIMUHeading)
	imuPlacer.run(visulizeCalibration)

	model = imuPlacer.getCalibratedModel()
	model.printToXML(constant_common.OPENSENSE_ASSET_PATH + 'calibrated_' + modelFileName)


def os_calibration_customized(seg2sens, os_model = 'Rajagopal_2015'):

	''' Apply calibration to the OpenSim model with rotation from segments to sensors '''
	
	model_fn = constant_common.OPENSENSE_ASSET_PATH + os_model + '_calibrated.osim'

	with open(model_fn, 'r') as f:
		data_xml = f.read()

	tree = BeautifulSoup(data_xml, 'xml')
	tree_model = tree.find('Model')
	tree_model['name'] = 'calibrated_' + os_model

	for segment in seg2sens.keys():
		imu_placement = tree.find('PhysicalOffsetFrame', {'name': constant_mt.MT_TO_OPENSENSE_MAP[segment]})
		imu_orientation = imu_placement.find('orientation')

		angle = R.from_matrix(seg2sens[segment]).as_euler('XYZ')
		imu_orientation.string = str(angle[0]) + ' ' + str(angle[1]) + ' ' + str(angle[2])

	with open(constant_common.OPENSENSE_ASSET_PATH + 'calibrated_' + os_model + '.osim', 'w') as f:
		f.write(tree.prettify())
		f.close()


def os_ik(orientationsFileName, os_model, visualizeTracking, catch_path):
	
	''' Perform IK with OpenSim/OpenSense '''

	startTime                   = 0
	endTime                     = 6000
	modelFileName               = os_model + '.osim'
	sensor_to_opensim_rotations = osim.Vec3(-math.pi/2, 0, 0)
	resultsDirectory            = catch_path

	imuIK = osim.IMUInverseKinematicsTool()
	imuIK.set_model_file(constant_common.OPENSENSE_ASSET_PATH + 'calibrated_' + modelFileName)
	imuIK.set_orientations_file(catch_path + orientationsFileName)
	imuIK.set_sensor_to_opensim_rotations(sensor_to_opensim_rotations)
	imuIK.set_results_directory(resultsDirectory)
	imuIK.set_time_range(0, startTime)
	imuIK.set_time_range(1, endTime)
	imuIK.run(visualizeTracking)


def get_all_ja_os(ik_fn, os_model, catch_path):
	
	''' Get all joint angles from OpenSim/OpenSense '''

	imu_os_ja = {}

	with open(catch_path + ik_fn, 'r') as f:
		txt    = f.readlines()
		header = txt[6].split('\t')

	angles = np.genfromtxt(catch_path + ik_fn, delimiter='\t', skip_header=7)
	dt     = pd.DataFrame(angles, columns = header)

	imu_os_ja['hip_adduction_l'] = 1*dt['hip_adduction_l'].to_numpy()
	imu_os_ja['hip_rotation_l']  = 1*dt['hip_rotation_l'].to_numpy()
	imu_os_ja['hip_flexion_l']   = 1*dt['hip_flexion_l'].to_numpy()
	imu_os_ja['knee_flexion_l']  = 1*dt['knee_angle_l'].to_numpy()
	imu_os_ja['ankle_angle_l']   = 1*dt['ankle_angle_l'].to_numpy()

	imu_os_ja['hip_adduction_r'] = 1*dt['hip_adduction_r'].to_numpy()
	imu_os_ja['hip_rotation_r']  = 1*dt['hip_rotation_r'].to_numpy()
	imu_os_ja['hip_flexion_r']   = 1*dt['hip_flexion_r'].to_numpy()
	imu_os_ja['knee_flexion_r']  = 1*dt['knee_angle_r'].to_numpy()
	imu_os_ja['ankle_angle_r']   = 1*dt['ankle_angle_r'].to_numpy()

	return imu_os_ja






