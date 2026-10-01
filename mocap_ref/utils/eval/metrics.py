# name: metrics.py
# description: compute metrics for evaluation


import math
import numpy as np 

from scipy.stats import pearsonr


def get_rmse(mocap, imu):

	''' Compute root-mean-square error (RMSE) between mocap- and IMU-based joint angles '''

	mse = np.nanmean(np.square(np.subtract(mocap, imu)))
	rmse = math.sqrt(mse)

	return rmse


def get_maxae(mocap, imu):

	''' Compute maximum absolute error (MaxAE) between mocap- and IMU-based joint angles '''

	mae = np.max(np.abs(np.subtract(mocap, imu)))

	return mae


def get_corrcoef(mocap, imu):
	
	''' Compute correlation coefficient (r) between mocap- and IMU-based joint angles '''
	
	corr_coef, _ = pearsonr(mocap, imu)

	return corr_coef


