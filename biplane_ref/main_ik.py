# name: main_ik.py 


import argparse


from scripts import run_mc10, run_mc10_cf, run_mocap
# run_mc10_opensense (needs opensim) is imported only when --do_opensense is used


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'HAKnee')
    parser.add_argument('--subject', type = str, default = None) # subject number, if not specified, run all subjects
    parser.add_argument('--task', type = str, default = None) # task name, if not specified, run all tasks
    parser.add_argument('--trial', type = int, default = None) # trial number, if not specified, run all trials (up to 3)
    parser.add_argument('--side', type = str, default = None) # side, if not specified, run all sides ('r' and 'l')

    parser.add_argument('--filter_type', type = str, default = 'VQF') # 'VQF', 'MAH', 'MAD', 'EKF', 'RIANN'
    parser.add_argument('--dim', type = str, default = '6d') # only '6d' is supported for MC10 IK

    parser.add_argument('--do_opensense', action = 'store_true') # whether to run OpenSense IK

    parser.add_argument('--do_cf', action = 'store_true') # whether to run IK with constraint feedback (NOTE: only applicable for VQF, EKF, MAD, and MAH) 
    parser.add_argument('--knee_gain', type = float, default = 0.9) # gain for knee constraint feedback, only applicable if --do_cf is specified
    parser.add_argument('--coupling', type = str, default = 'walker', choices = ['walker', 'reuben', 'healthy_global', 'healthy_loso']) # knee coupling curve for --do_cf

    parser.add_argument('--do_mocap', action = 'store_true') # whether to run mocap IK

    args = parser.parse_args()

    if args.do_mocap:
        run_mocap.mocap_ik_main(args.dataset, args.subject, args.task, args.trial, args.side)

    else:
        if args.do_opensense:
            from scripts import run_mc10_opensense
            run_mc10_opensense.mc10_opensense_ik_main(args.dataset, args.subject, args.task, args.trial, args.side, args.filter_type, args.dim)
        elif args.do_cf:
            run_mc10_cf.mc10_ik_cf_main(args.dataset, args.subject, args.task, args.trial, args.side, args.filter_type, args.dim, tuning = False, knee_gain = args.knee_gain, coupling = args.coupling)
        else:
            run_mc10.mc10_ik_main(args.dataset, args.subject, args.task, args.trial, args.side, args.filter_type, args.dim)


if __name__ == '__main__':
    main()






