"""
Reconstructs the full training command for the search winner of one
(model, dataset) pair, selected by mean validation accuracy from the
runs_*.csv files written by submit_gt_search.sbatch.

Prints the command to stdout (nothing else), so sbatch scripts can do:
    CMD=$(python gt_confirm_cmd.py --model polynormer --dataset cora ...)
    eval "$CMD"

The fixed (non-searched) flags below MUST stay in sync with the search grids
in submit_gt_search.sbatch.
"""

import argparse
import glob
import os
import sys

import pandas as pd


def dataset_flags(dataset):
    """Protocol flags per dataset — mirrors submit_gt_search.sbatch."""
    if dataset in ('cora', 'citeseer', 'pubmed'):
        return '--rand_split_class --valid_num 500 --test_num 1000 --seed 123'
    if dataset in ('minesweeper', 'questions'):
        return '--metric rocauc'
    return ''


def dataset_epochs(dataset):
    if dataset in ('roman-empire', 'amazon-ratings'):
        return 2500
    if dataset in ('chameleon', 'squirrel'):
        return 500
    return 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['polynormer', 'sgformer'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--result_dir', default='gt_search/results')
    ap.add_argument('--out_result_dir', default='results_tuned')
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--model_dir', default='models')
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.result_dir, args.dataset, 'runs_*.csv'))
    if not files:
        sys.exit(f'ERROR: no runs CSVs for {args.dataset} under {args.result_dir}')
    df = pd.concat([pd.read_csv(f, on_bad_lines='skip') for f in files],
                   ignore_index=True)
    df = df[(df['model'] == args.model) & (df['reg_type'] == 'none')]
    df['best_valid'] = pd.to_numeric(df['best_valid'], errors='coerce')
    df = df.dropna(subset=['best_valid', 'config'])
    df = df.drop_duplicates(subset=['config', 'run'], keep='last')  # requeue safety
    if df.empty:
        sys.exit(f'ERROR: no baseline search rows for {args.model}/{args.dataset}')

    winner = df.groupby('config')['best_valid'].mean().idxmax()
    cfg = dict(kv.split('=', 1) for kv in winner.split(','))

    common = (f"python main.py --model {args.model} --dataset {args.dataset} "
              f"--lr {cfg['lr']} --hidden_channels {cfg['hid']} "
              f"--dropout {cfg['do']} --weight_decay {cfg['wd']} "
              f"--num_heads {cfg['heads']} "
              f"--runs {args.runs} --device {args.device} "
              f"--result_dir {args.out_result_dir} --model_dir {args.model_dir} "
              f"--paired_seeds {dataset_flags(args.dataset)}")

    if args.model == 'polynormer':
        # in_dropout 0.2 is fixed in the search grid
        cmd = (f"{common} --local_layers {cfg['ll']} --global_layers {cfg['gl']} "
               f"--local_epochs {cfg['le']} --global_epochs {cfg['ge']} "
               f"--beta {cfg['beta']} --in_dropout 0.2 --save_model")
    else:
        cmd = (f"{common} --layers {cfg['gcn_l']} --tr_layers {cfg['tr_l']} "
               f"--graph_weight {cfg['gw']} --alpha {cfg['alpha']} "
               f"--use_graph --use_residual --epochs {dataset_epochs(args.dataset)}")
        if cfg.get('tr_do') not in (None, '', 'None'):
            cmd += f" --tr_dropout {cfg['tr_do']}"
        if cfg.get('tr_wd') not in (None, '', 'None'):
            cmd += f" --tr_weight_decay {cfg['tr_wd']}"
        if cfg.get('bn') == 'True':
            cmd += ' --use_bn'

    print(cmd)


if __name__ == '__main__':
    main()
