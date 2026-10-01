# name: eval.py
# description: evaluate kinematics compared to the mocap-based reference


import argparse

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from utils.eval.run_eval import evaluate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--f_type', type = str, default = None) # 'Xsens', 'MAH', 'VQF', 'MAD', or 'EKF'
    parser.add_argument('--dim', type = str, default = '9d') # '9d' or '6d'
    parser.add_argument('--reference', type = str, default = 'direct') # 'direct' or 'opensim
    parser.add_argument('--selected_setup', type = str, default = 'mm') # sensor placement, i.e., 'mm' (for main analysis), 'hh', 'll', or 'ff'

    parser.add_argument('--enable_opensense', action = 'store_true') # enable OpenSense evaluation
    parser.add_argument('--enable_cf', action = 'store_true') # enable constraint-feedback evaluation

    parser.add_argument('--subject', type = int, default = None) # subject number, if not specified, run all subjects
    parser.add_argument('--task', type = str, default = None) # task being performed, if not specified, run all tasks

    parser.add_argument('--enable_mocap_alignment', action = 'store_true') # enable mocap alignment

    parser.add_argument('--enable_drift_eval', action = 'store_true') # enable drift evaluation (i.e., evaluate the RMSE of the first few cycles and the last few cycles separately to show drifting)

    parser.add_argument('--enable_psa', action = 'store_true') # enable perfect standing alignment for 9D data (PSA: perfect standing alignment)


    args = parser.parse_args()

    evaluate(args.f_type, args.dim, args.subject, args.task, args.reference, args.enable_mocap_alignment, args.selected_setup, args.enable_opensense, args.enable_cf, enable_psa = args.enable_psa, enable_drift_eval = args.enable_drift_eval)


if __name__ == '__main__':
    main()



