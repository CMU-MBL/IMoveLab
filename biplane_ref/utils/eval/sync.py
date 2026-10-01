# name: sync.py
# description: supporting functions for syncing


import numpy as np
from copy import deepcopy



def sync_data(ja, lag, ja_length):

    ''' Sync the data based on the lag '''

    synced_ja = {}
    for joint in ja.keys():
        synced_ja[joint] = ja[joint][lag:lag + ja_length]

    return synced_ja


def apply_sync(ja1, ja2, lag):

    ''' Sync the data based on the lag '''

    ja1_new = deepcopy(ja1)
    ja2_new = deepcopy(ja2)
    if lag >= 0:
        ja_length = len(ja2[list(ja2.keys())[0]])
        ja1_new   = sync_data(ja1, lag, ja_length)

    elif lag < 0:
        ja_length = len(ja1[list(ja1.keys())[0]])
        ja2_new   = sync_data(ja2, -lag, ja_length)

    return ja1_new, ja2_new


def find_best_match(long_sig, short_sig, nan_flag = False):

    ''' Another way to find lag '''
    
    x = np.asarray(long_sig, dtype=float)
    q = np.asarray(short_sig, dtype=float)

    if nan_flag:
        q = q[~np.isnan(q)]  # remove NaN values from short_sig
        
    N, M = x.size, q.size
    if M > N:
        raise ValueError("short_sig must not be longer than long_sig")

    q_mean = q.mean()
    q_std = q.std()
    
    if q_std == 0:
        raise ValueError("short_sig has zero variance")

    csum = np.cumsum(np.r_[0.0, x])           
    csum2 = np.cumsum(np.r_[0.0, x*x])  
    win_sum  = csum[M:]  - csum[:-M]             
    win_sum2 = csum2[M:] - csum2[:-M]        
    win_mean = win_sum / M
    
    win_var = np.maximum(win_sum2 / M - win_mean**2, 0.0)
    win_std = np.sqrt(win_var)
    
    valid = win_std > 0

    cross = np.correlate(x, q, mode='valid') 

    ncc = np.full_like(cross, fill_value=np.nan, dtype=float)
    numerator = cross - M * win_mean * q_mean
    denom = (M * win_std * q_std)
    ncc[valid] = numerator[valid] / denom[valid]

    best_idx = int(np.nanargmax(ncc))
    
    
    return best_idx





