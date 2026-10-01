# name: sfa.py
# description: apply sensor fusion algorithms for estimating orientation


from ahrs.filters import Mahony, Madgwick, EKF
from vqf import PyVQF
from riann.riann import RIANN
import numpy as np


# --- Apply VQF --- #
def apply_vqf(gyr, acc, mag, dim = '9D', fs = 100, params = None):

    ''' Apply VQF to get orientation '''

    if params is None:
        vqf = PyVQF(gyrTs = 1.0/fs)
    else:
        vqf = PyVQF(gyrTs = 1.0/fs, tauAcc = params[0], tauMag = params[1])

    if dim == '6D':
        temp_estimation = vqf.updateBatch(gyr, acc)
    else:
        temp_estimation = vqf.updateBatch(gyr, acc, mag)

    return temp_estimation


# --- Apply Madgwick --- #
def apply_madgwick(gyr, acc, mag, dim = '9D', fs = 100, params = None):

    ''' Apply Madgwick to get orientation '''

    if dim == '6D':
        if params is None:
            temp_estimation = Madgwick(gyr = gyr, acc = acc, frequency = fs)
        else:
            temp_estimation = Madgwick(gyr = gyr, acc = acc, frequency = fs, gain = params[0])
            
    else:
        if params is None:
            temp_estimation = Madgwick(gyr = gyr, acc = acc, mag = mag, frequency = fs)
        else:
            temp_estimation = Madgwick(gyr = gyr, acc = acc, mag = mag, frequency = fs, gain = params[0])

    return temp_estimation


# --- Apply Mahony --- #
def apply_mahony(gyr, acc, mag, dim = '9D', fs = 100, params = None):

    ''' Apply Mahony to get orientation '''

    if dim == '6D':
        if params is None:
            temp_estimation = Mahony(gyr = gyr, acc = acc, frequency = fs)
        else:
            temp_estimation = Mahony(gyr = gyr, acc = acc, frequency = fs, k_P = params[0], k_I = params[1])

    else:
        if params is None:
            temp_estimation = Mahony(gyr = gyr, acc = acc, mag = mag, frequency = fs)
        else:
            temp_estimation = Mahony(gyr = gyr, acc = acc, mag = mag, frequency = fs, k_P = params[0], k_I = params[1])

    return temp_estimation


# --- Apply EKF --- #
def apply_ekf(gyr, acc, mag, dim = '9D', fs = 100, params = None):

    ''' Apply EKF to get orientation '''

    if dim == '6D':
        if params is None:
            temp_estimation = EKF(gyr = gyr, acc = acc, frequency = fs, frame = 'ENU')
        else:
            temp_estimation = EKF(gyr = gyr, acc = acc, frequency = fs, frame = 'ENU', noises = [params[0]**2, params[1]**2, params[2]**2])

    else:
        if params is None:
            temp_estimation = EKF(gyr = gyr, acc = acc, mag = mag, frequency = fs, frame = 'ENU')
        else:
            temp_estimation = EKF(gyr = gyr, acc = acc, mag = mag, frequency = fs, frame = 'ENU', noises = [params[0]**2, params[1]**2, params[2]**2])

    return temp_estimation


# --- Apply RIANN --- #
def apply_riann(gyr, acc, fs = 100):

    ''' Apply RIANN to get sensor orientation '''

    riann = RIANN()

    temp_estimation = riann.predict(acc, gyr, fs)

    return temp_estimation
