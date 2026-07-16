"""
Selects the best hyperparameter config per (dataset, model) from the search
runs written by submit_gt_search.sbatch.

Selection is by MEAN VALIDATION accuracy across the search runs (test is
reported but never used for selection). The chosen config's fingerprint is
printed so it can be confirmed with a 10-run job before use as a baseline.

Usage:
    python analyze_gt_search.py [--result_dir gt_search/results] [--top 5]
"""

import argparse
import glob
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_dir', type=str, default='gt_search/results')
    ap.add_argument('--top', type=int, default=5)
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.result_dir, '*', 'runs_*.csv'))
    if not files:
        raise SystemExit(f'no runs_*.csv found under {args.result_dir}')

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df['reg_type'] == 'none']

    best_rows = []
    for (dataset, model), g in df.groupby(['dataset', 'model']):
        agg = g.groupby('config').agg(
            val_mean=('best_valid', 'mean'),
            val_std=('best_valid', 'std'),
            test_mean=('test_at_best_valid', 'mean'),
            test_std=('test_at_best_valid', 'std'),
            n_runs=('run', 'count'),
        ).sort_values('val_mean', ascending=False)

        print(f'\n=== {dataset} / {model} — top {args.top} of {len(agg)} configs '
              f'(selected by mean validation) ===')
        print(agg.head(args.top).round(2).to_string())

        top = agg.iloc[0]
        best_rows.append({
            'dataset': dataset, 'model': model,
            'best_config': agg.index[0],
            'val': f"{top['val_mean']:.2f} ± {top['val_std']:.2f}",
            'test': f"{top['test_mean']:.2f} ± {top['test_std']:.2f}",
            'n_runs': int(top['n_runs']),
        })

    out = pd.DataFrame(best_rows).sort_values(['model', 'dataset'])
    print('\n\n=== Selected configs (confirm each with 10 runs before use) ===')
    print(out.to_string(index=False))
    out_path = os.path.join(args.result_dir, 'selected_configs.csv')
    out.to_csv(out_path, index=False)
    print(f'\nWritten to {out_path}')


if __name__ == '__main__':
    main()
