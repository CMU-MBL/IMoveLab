# name: main_eval.py


import argparse


from scripts import run_eval, run_eval_cf


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


    args = parser.parse_args()
    
    if args.do_cf:
        run_eval_cf.eval_cf_main(dataset = args.dataset, subject = args.subject, task = args.task, trial = args.trial, side = args.side, filter_type = args.filter_type, dim = args.dim, tuning = False, knee_gain = args.knee_gain, coupling = args.coupling)

    else:
        run_eval.eval_main(dataset = args.dataset, subject = args.subject, task = args.task, trial = args.trial, side = args.side, filter_type = args.filter_type, dim = args.dim, tuning = False, do_opensense = args.do_opensense)


if __name__ == '__main__':
    main()






