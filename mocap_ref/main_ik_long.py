# name: main_ik_long.py
# description: IMU-based IK for MTw data collected during long trials 


import argparse

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from utils import common
from scripts import run_mt, run_mocap, run_mt_opensense, run_mt_cf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_type', type = str, default = None) # 'Xsens', 'MAH', 'VQF', 'MAD', or 'EKF'
    parser.add_argument('--dim', type = str, default = '9d') # '9d' or '6d'

    parser.add_argument('--do_mocap', action = 'store_true') # run mocap

    parser.add_argument('--do_opensense', action = 'store_true') # run OpenSense for IMU data   
    parser.add_argument('--do_constraint_feedback', action = 'store_true') # run constraint-feedback method for IMU data (6D only)

    parser.add_argument('--subject', type = str, default = None) # subject number, if not specified, run all subjects
    parser.add_argument('--task', type = str, default = None) # long_walk1, long_walk2, or long_walk3

    args = parser.parse_args()

    if args.do_mocap:
        run_mocap.mocap_ik(args.subject, args.task, source = 'mt_long')

    else:
        selected_setup = 'mm'

        if args.f_type is None:
            filter_config_check = True
        else:
            filter_config_check, error_msg = common.check_filter_config(args.f_type, args.dim)

        if filter_config_check:
            if args.do_opensense:
                run_mt_opensense.mt_ik_opensense(selected_setup, args.f_type, args.dim, args.subject, 'long_walk1', source = 'mt_long')

            else:
                if args.do_constraint_feedback:
                    run_mt_cf.mt_ik(selected_setup, args.f_type, args.dim, args.subject, 'long_walk1', source = 'mt_long')
                else:
                    run_mt.mt_ik(selected_setup, args.f_type, args.dim, args.subject, 'long_walk1', source = 'mt_long') 
        else:
            print(error_msg)


if __name__ == '__main__':
    main()







