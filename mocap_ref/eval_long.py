# name: eval_long.py
# description: evaluate kinematics compared to the mocap-based reference during hour-long trials


import argparse

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from utils.eval.run_eval_long import evaluate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--f_type', type = str, default = 'VQF') # 'Xsens', 'MAH', 'VQF', 'MAD', or 'EKF'
    parser.add_argument('--dim', type = str, default = '9d') # '9d' or '6d'
    parser.add_argument('--reference', type = str, default = 'direct') # 'direct' or 'opensim'
    parser.add_argument('--selected_setup', type = str, default = 'mm') # sensor placement, i.e., 'mm' (for main analysis), 'hh', 'll', or 'ff'
    parser.add_argument('--eval_mode', type = str, default = 'trial_rmsd') # 'trial_rmsd' or 'minute_rmsd' or 'visualization'

    parser.add_argument('--enable_opensense', action = 'store_true') # enable OpenSense for mocap reference
    parser.add_argument('--enable_cf', action = 'store_true') # enable constraint-feedback method for IMU data (6D only)

    parser.add_argument('--subject', type = str, default = None) # subject number, if not specified, run all subjects

    parser.add_argument('--enable_mocap_alignment', action = 'store_true') # enable mocap alignment
    parser.add_argument('--enable_mocap', action = 'store_true') # enable mocap-based NOTE: only for visualization mode


    args = parser.parse_args()

    evaluate(args.f_type, args.dim, args.subject, args.reference, args.enable_mocap_alignment, args.selected_setup, args.enable_opensense, args.enable_cf, args.eval_mode, args.enable_mocap)


if __name__ == '__main__':
    main()




