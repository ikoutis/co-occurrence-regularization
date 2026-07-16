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
    ap.add_argument('--result_dir', type=str, default='results/placebo')
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.result_dir, '*', 'runs_*.csv'))
    if not files:
        raise SystemExit(f'no runs_*.csv found under {args.result_dir}')

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['penalty_transform'] = df['penalty_transform'].fillna('')

    rows = []
    for (dataset, model), g in df.groupby(['dataset', 'model']):
        base = g[g['reg_type'] == 'none']
        if base.empty:
            print(f'WARNING: no baseline rows for {dataset}/{model}, skipping')
            continue
        # deduplicate baseline (keep last occurrence per run index)
        base = base.drop_duplicates(subset='run', keep='last').set_index('run')
        base_mean, base_std = base['test_at_best_valid'].mean(), base['test_at_best_valid'].std()

        for (reg_type, transform), cond in g[g['reg_type'] != 'none'].groupby(
                ['reg_type', 'penalty_transform']):
            # validation-based lambda selection
            val_by_lambda = cond.groupby('lambda')['best_valid'].mean()
            lam = val_by_lambda.idxmax()
            sel = cond[cond['lambda'] == lam].drop_duplicates(
                subset='run', keep='last').set_index('run')

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
                'condition': f'{reg_type}/{transform or "none"}',
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

    out = pd.DataFrame(rows).sort_values(['dataset', 'model', 'condition'])
    print(out.to_markdown(index=False))
    out_path = os.path.join(args.result_dir, 'placebo_summary.md')
    with open(out_path, 'w') as f:
        f.write('# Placebo experiment summary\n\n')
        f.write('λ selected by mean validation accuracy; test reported at λ*.\n')
        f.write('penalty_dist ≈ 0 ⇒ the transform barely changed the penalty '
                'and that placebo row has no power.\n\n')
        f.write(out.to_markdown(index=False))
        f.write('\n')
    print(f'\nWritten to {out_path}')


if __name__ == '__main__':
    main()
