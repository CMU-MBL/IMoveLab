# name: sfa.py
# description: sensor fusion


from ahrs.filters import Mahony, Madgwick, EKF, UKF
from vqf import PyVQF
from riann.riann import RIANN



def apply_vqf(gyr, acc, fs = 100, params = None):

    ''' apply VQF to get sensor orientation (quaternion) '''

    if params is None:
        vqf = PyVQF(gyrTs = 1.0/fs)
        
    else:
        vqf = PyVQF(gyrTs = 1.0/fs, tauAcc = params[0], tauMag = params[1])

    temp_estimate = vqf.updateBatch(gyr, acc)


    return temp_estimate


def apply_mahony(gyr, acc, fs = 100, params = None):

    ''' apply Mahony filter to get sensor orientation (quaternion) '''
    
    if params is None:
        temp_estimate = Mahony(gyr = gyr, acc = acc, frequency = fs)

    else:
        temp_estimate = Mahony(gyr = gyr, acc = acc, frequency = fs, k_P = params[0], k_I = params[1])


    return temp_estimate


def apply_madgwick(gyr, acc, fs = 100, params = None):

    ''' apply Madgwick filter to get sensor orientation (quaternion) '''
    
    if params is None:
        temp_estimate = Madgwick(gyr = gyr, acc = acc, frequency = fs)

    else:
        temp_estimate = Madgwick(gyr = gyr, acc = acc, frequency = fs, gain = params[0])


    return temp_estimate


def apply_ekf(gyr, acc, fs = 100, params = None):

    ''' apply EKF to get sensor orientation (quaternion) '''
    
    if params is None:
        temp_estimate = EKF(gyr = gyr, acc = acc, frequency = fs, frame = 'ENU')

    else:
        temp_estimate = EKF(gyr = gyr, acc = acc, frequency = fs, frame = 'ENU', noises = [params[0]**2, params[1]**2, params[2]**2])


    return temp_estimate


def apply_ukf(gyr, acc, fs = 100, params = None):

    ''' apply UKF to get sensor orientation (quaternion) '''

    if params is None:
        temp_estimate = UKF(gyr = gyr, acc = acc, frequency = fs)

    else:
        print('Applying UKF with alpha = {}, beta = {}, kappa = {} ...'.format(params[0], params[1], params[2]))
        temp_estimate = UKF(gyr = gyr, acc = acc, frequency = fs, alpha = params[0], beta = params[1], kappa = params[2])


    return temp_estimate


def apply_riann(gyr, acc, fs = 100):

    ''' apply RIANN to get sensor orientation (quaternion) '''

    riann = RIANN()

    temp_estimate = riann.predict(acc, gyr, fs)


    return temp_estimate





