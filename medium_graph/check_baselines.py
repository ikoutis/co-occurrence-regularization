"""
Automated baseline acceptance gate (EXPERIMENT_PLAN E1.1/E1.2).

Reads baseline rows (reg_type == 'none') from runs_*.csv under the given
result dirs and checks:

1. GT published targets: the 13 published-config (model, dataset) combos
   must land within TOLERANCE of the numbers from the official
   Polynormer repo / SGFormer paper (see GT_BASELINES_README.md).
2. Degeneracy heuristic (all models incl. GNNs): a baseline more than
   DEGEN_GAP points below the best baseline of any model on the same
   dataset is flagged — this catches roman-empire/GCN-at-28.5-style
   broken configs that have no published target line.

Exit code 1 if any FAIL, so it can gate follow-up submissions.

Usage:
    python check_baselines.py --result_dir results/placebo results/unified
    python check_baselines.py --result_dir GTs_baselines/results_tuned
"""

import argparse
import glob
import os
import sys

import pandas as pd

TOLERANCE = 1.5   # points from published target -> FAIL beyond this
DEGEN_GAP = 8.0   # points below same-dataset best baseline -> FAIL

# Published numbers: Polynormer official-repo README ("expected results",
# single run); SGFormer paper (20-per-class splits). ROC-AUC where noted.
GT_TARGETS = {
    ('polynormer', 'roman-empire'): 92.48,
    ('polynormer', 'amazon-ratings'): 55.04,
    ('polynormer', 'minesweeper'): 97.19,
    ('polynormer', 'questions'): 78.35,
    ('polynormer', 'amazon-computer'): 94.07,
    ('polynormer', 'amazon-photo'): 96.67,
    ('polynormer', 'coauthor-cs'): 95.28,
    ('polynormer', 'coauthor-physics'): 97.14,
    ('polynormer', 'wikics'): 81.20,
    ('sgformer', 'cora'): 84.5,
    ('sgformer', 'citeseer'): 72.6,
    ('sgformer', 'pubmed'): 80.3,
    # sgformer/chameleon: published ~44.9 on the filtered dataset; both
    # repos load filtered, but the official run uses --method ours with a
    # slightly different eval protocol — treat as informational
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_dir', type=str, nargs='+', required=True)
    args = ap.parse_args()

    files = [f for d in args.result_dir
             for f in glob.glob(os.path.join(d, '*', 'runs_*.csv'))]
    if not files:
        raise SystemExit(f'no runs_*.csv found under {args.result_dir}')

    df = pd.concat([pd.read_csv(f, on_bad_lines='skip') for f in files],
                   ignore_index=True)
    df = df[df['dataset'] != 'dataset']
    df = df[df['reg_type'] == 'none']
    for col in ('best_valid', 'test_at_best_valid', 'run', 'seed'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['test_at_best_valid', 'run'])
    if 'config' not in df.columns:
        df['config'] = ''
    df = df.drop_duplicates(subset=['dataset', 'model', 'config', 'seed', 'run'],
                            keep='last')

    agg = df.groupby(['dataset', 'model']).agg(
        test_mean=('test_at_best_valid', 'mean'),
        test_std=('test_at_best_valid', 'std'),
        n=('run', 'count')).reset_index()

    failures = 0
    print(f"{'dataset':<18} {'model':<14} {'baseline':<16} {'target':<8} verdict")
    print('-' * 70)
    for _, row in agg.sort_values(['dataset', 'model']).iterrows():
        model_key = row['model'].lower()
        # normalize MPNN_gcn -> gcn etc. for the degeneracy grouping only
        target = GT_TARGETS.get((model_key, row['dataset']))
        best_on_dataset = agg[agg['dataset'] == row['dataset']]['test_mean'].max()
        verdict, detail = 'ok', ''
        if target is not None:
            gap = row['test_mean'] - target
            detail = f'{target:<8.2f}'
            if gap < -TOLERANCE:
                verdict, failures = f'FAIL ({gap:+.2f} vs published)', failures + 1
            else:
                verdict = f'PASS ({gap:+.2f})'
        else:
            detail = f'{"—":<8}'
        if row['test_mean'] < best_on_dataset - DEGEN_GAP:
            verdict += f'  DEGENERATE? ({best_on_dataset - row["test_mean"]:.1f} pts below best model on this dataset)'
            failures += 1
        print(f"{row['dataset']:<18} {row['model']:<14} "
              f"{row['test_mean']:.2f} ± {row['test_std']:.2f} (n={int(row['n'])})"
              f"  {detail} {verdict}")

    print('-' * 70)
    if failures:
        print(f'{failures} baseline(s) FAILED the gate — investigate before '
              f'interpreting any regularization delta on them.')
        sys.exit(1)
    print('All baselines pass.')


if __name__ == '__main__':
    main()
