# name: mc10_event.py
# description: gait event detection for the MC10 Biostamp data


import numpy as np
import pandas as pd

from scipy import signal


# --- Filter data before identifying gait cycles or exercise repetitions --- #
def filter_signal(input_signal, fs, cutoff = 15):

    ''' Filter data before identifying gait cycles or exercise repetitions '''

    Wn   = cutoff*2/fs 
    b, a = signal.butter(4, Wn, btype = 'low')

    filtered_signal = signal.filtfilt(b, a, input_signal)

    return filtered_signal


# --- Detect gait events during walking --- #
def detect_gait_events(lateral_shank_vel, fs = 100):

    ''' Obtain gait events, only mid swing from IMU data '''

    min_peak_distance = 0.5*fs 
    
    filtered_lateral_shank_vel = filter_signal(lateral_shank_vel, fs)

    mid_swing_index, mid_swing_value = signal.find_peaks(filtered_lateral_shank_vel, height = [2, 10], distance = min_peak_distance)
    mid_swing_value                  = mid_swing_value['peak_heights']

    gait_events = {}
    gait_events['ms_index'] = 1*mid_swing_index
    gait_events['ms_value'] = 1*mid_swing_value
    
    return gait_events











