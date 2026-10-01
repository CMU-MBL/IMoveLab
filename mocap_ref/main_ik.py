# name: main_ik.py
# description: main file for inverse kinematics


import argparse

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from utils import common
from scripts import run_mt_opensense, run_mt, run_mocap, run_mt_cf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_setup', type = str, default = 'mm') # sensor placement, i.e., 'mm' (for main analysis), 'hh', 'll', or 'ff'
    parser.add_argument('--f_type', type = str, default = None) # 'Xsens', 'MAH', 'VQF', 'MAD', or 'EKF'
    parser.add_argument('--dim', type = str, default = '9d') # '9d' or '6d'

    parser.add_argument('--do_mocap', action = 'store_true') # run mocap

    parser.add_argument('--do_opensense', action = 'store_true') # run OpenSense for IMU data
    parser.add_argument('--do_constraint_feedback', action = 'store_true') # run constraint-feedback method for IMU data (6D only)

    parser.add_argument('--subject', type = int, default = None) # subject number, if not specified, run all subjects
    parser.add_argument('--task', type = str, default = None) # task being performed, if not specified, run all tasks

    args = parser.parse_args()

    if args.do_mocap:
        run_mocap.mocap_ik(args.subject, args.task)

    else:
        if args.f_type is None:
            filter_config_check = True
        else:
            filter_config_check, error_msg = common.check_filter_config(args.f_type, args.dim)

        if filter_config_check:
            if args.do_opensense:
                run_mt_opensense.mt_ik_opensense(args.selected_setup, args.f_type, args.dim, args.subject, args.task) 

            else:
                if args.do_constraint_feedback:
                    run_mt_cf.mt_ik(args.selected_setup, args.f_type, args.dim, args.subject, args.task)  
                else:
                    run_mt.mt_ik(args.selected_setup, args.f_type, args.dim, args.subject, args.task) 
        else:
            print(error_msg)


if __name__ == '__main__':
    main()





