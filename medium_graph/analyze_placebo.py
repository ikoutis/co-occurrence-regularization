"""
Analysis for the placebo experiment (see PLACEBO_README.md).

Reads the per-run CSVs written by save_runs_detail (runs_*.csv), performs
validation-based lambda selection per condition, and compares each condition
against the paired baseline with a Wilcoxon signed-rank test.

Protocol:
  1. For each (dataset, model, reg_type, penalty_transform): pick lambda* by
     highest MEAN VALIDATION accuracy across runs. Test accuracy is never
     used for selection.
  2. At lambda*, pair each run's test accuracy with the baseline run of the
     same index (runs are paired via --paired_seeds: same split, same init,
     same training noise stream).
  3. Report mean +/- std, paired delta, Wilcoxon p-value, and the placebo
     distinguishability diagnostics (penalty_dist, offdiag_cv). A shuffle
     row with penalty_dist near 0 has no statistical power regardless of
     its p-value — the transform barely changed the penalty.

Usage:
    python analyze_placebo.py [--result_dir results/placebo]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def paired_pvalue(deltas):
    deltas = np.asarray(deltas, dtype=float)
    nonzero = deltas[deltas != 0]
    if len(nonzero) < 5:
        return float('nan')
    if HAVE_SCIPY:
        try:
            return wilcoxon(nonzero).pvalue
        except ValueError:
            return float('nan')
    # sign-test fallback
    from math import comb
    n, k = len(nonzero), int((nonzero > 0).sum())
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_dir', type=str, nargs='+', default=['results/placebo'],
                    help='one or more result dirs to merge (e.g. results/placebo '
                         'results/unified for the full E2.1 table)')
    args = ap.parse_args()

    files = [f for d in args.result_dir
             for f in glob.glob(os.path.join(d, '*', 'runs_*.csv'))]
    if not files:
        raise SystemExit(f'no runs_*.csv found under {args.result_dir}')

    df = pd.concat([pd.read_csv(f, on_bad_lines='skip') for f in files],
                   ignore_index=True)
    # guard against interleaved/duplicate header rows from concurrent appends
    df = df[df['dataset'] != 'dataset']
    for col in ('best_valid', 'test_at_best_valid', 'lambda', 'run', 'seed'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['best_valid', 'test_at_best_valid', 'run'])
    df['penalty_transform'] = df['penalty_transform'].fillna('')
    if 'mlp_epochs' not in df.columns:
        df['mlp_epochs'] = ''
    df['mlp_epochs'] = df['mlp_epochs'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True)
    if 'config' not in df.columns:
        df['config'] = ''
    df['config'] = df['config'].fillna('')

    rows = []
    for (dataset, model), g in df.groupby(['dataset', 'model']):
        if g['config'].nunique() > 1:
            print(f'WARNING: {dataset}/{model} has rows from {g["config"].nunique()} '
                  f'different hyperparameter configs in these result dirs — '
                  f'selection/pairing may mix stale rows. Configs:')
            for cfg in g['config'].unique():
                print(f'    {cfg}')
        base = g[g['reg_type'] == 'none']
        if base.empty:
            print(f'WARNING: no baseline rows for {dataset}/{model}, skipping')
            continue
        # deduplicate (keep last occurrence per seed+run: requeued tasks
        # re-append rows) and key the pairing on (seed, run), not run alone
        base = base.drop_duplicates(subset=['seed', 'run'], keep='last').set_index(['seed', 'run'])
        base_mean, base_std = base['test_at_best_valid'].mean(), base['test_at_best_valid'].std()

        for (reg_type, transform, mlp_ep), cond in g[g['reg_type'] != 'none'].groupby(
                ['reg_type', 'penalty_transform', 'mlp_epochs']):
            # dedup BEFORE lambda selection: requeue re-appends can otherwise
            # give some lambdas double weight in the validation mean
            cond = cond.drop_duplicates(subset=['lambda', 'seed', 'run'], keep='last')
            # validation-based lambda selection
            val_by_lambda = cond.groupby('lambda')['best_valid'].mean()
            lam = val_by_lambda.idxmax()
            sel = cond[cond['lambda'] == lam].drop_duplicates(
                subset=['seed', 'run'], keep='last').set_index(['seed', 'run'])

            common = sel.index.intersection(base.index)
            if len(common) == 0:
                print(f'WARNING: no paired runs for {dataset}/{model}/{reg_type}/{transform}')
                continue
            deltas = (sel.loc[common, 'test_at_best_valid']
                      - base.loc[common, 'test_at_best_valid']).values

            pdist = pd.to_numeric(sel['penalty_dist'], errors='coerce').mean()
            pcv = pd.to_numeric(sel['offdiag_cv'], errors='coerce').mean()

            rows.append({
                'dataset': dataset,
                'model': model,
                'condition': f'{reg_type}/{transform or "none"}'
                             + (f'/mlp_ep{mlp_ep}' if mlp_ep not in ('', '500') else ''),
                'lambda*': lam,
                'n_pairs': len(common),
                'baseline': f'{base_mean:.2f} ± {base_std:.2f}',
                'test': f"{sel.loc[common, 'test_at_best_valid'].mean():.2f} "
                        f"± {sel.loc[common, 'test_at_best_valid'].std():.2f}",
                'paired_Δ': f'{deltas.mean():+.2f}',
                'p (Wilcoxon)': f'{paired_pvalue(deltas):.3f}',
                'penalty_dist': f'{pdist:.3f}' if pd.notna(pdist) else '—',
                'offdiag_cv': f'{pcv:.3f}' if pd.notna(pcv) else '—',
            })

    if not rows:
        raise SystemExit('No comparable (baseline, condition) pairs found yet — '
                         'is the sweep still running?')
    out = pd.DataFrame(rows).sort_values(['dataset', 'model', 'condition'])
    try:
        table = out.to_markdown(index=False)  # needs 'tabulate'
    except ImportError:
        table = out.to_string(index=False)
    print(table)
    out_path = os.path.join(args.result_dir[0], 'placebo_summary.md')
    with open(out_path, 'w') as f:
        f.write('# Placebo experiment summary\n\n')
        f.write('λ selected by mean validation accuracy; test reported at λ*.\n')
        f.write('penalty_dist ≈ 0 ⇒ the transform barely changed the penalty '
                'and that placebo row has no power.\n\n')
        f.write(table)
        f.write('\n')
    print(f'\nWritten to {out_path}')


if __name__ == '__main__':
    main()
