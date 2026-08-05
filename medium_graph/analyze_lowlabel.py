"""
Analysis for the low-label experiment (E3.1) — see LOWLABEL_README.md.

For each (dataset, model, budget, condition): validation-selected lambda,
paired delta vs the SAME-BUDGET baseline, Wilcoxon p, and the estimator
covariates (distance of the penalty source to the all-label oracle matrix,
number of train-train edges for the count estimator, MLP accuracy).

Output: one accuracy-vs-budget curve table per (dataset, model), plus a
compact significant-effects list. Conditions:
    count  — train-train edge counting (leakage-free)
    mlp    — features-only MLP estimator (leakage-free)
    oracle — all-label matrix (LEAKY upper bound, diagnostic only)
    mlp/shuffle — placebo

Each result dir is analyzed SEPARATELY (the matched-validation run uses a
different selection protocol and must never be pooled with the main run).

Usage:
    python analyze_lowlabel.py [--result_dir results/lowlabel]
    python analyze_lowlabel.py --result_dir results/lowlabel results/lowlabel_mv
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
    from math import comb
    n, k = len(nonzero), int((nonzero > 0).sum())
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(dirs):
    files = [f for d in dirs for f in glob.glob(os.path.join(d, '*', 'runs_*.csv'))]
    if not files:
        raise SystemExit(f'no runs_*.csv found under {dirs}')
    df = pd.concat([pd.read_csv(f, on_bad_lines='skip') for f in files],
                   ignore_index=True)
    df = df[df['dataset'] != 'dataset']
    for c in ('best_valid', 'test_at_best_valid', 'lambda', 'run', 'seed', 'budget'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['test_at_best_valid', 'run', 'budget'])
    df['penalty_transform'] = df.get('penalty_transform', '').fillna('')
    return df


def analyze_dir(result_dir):
    df = load([result_dir])
    print(f'\n################ {result_dir} ################')

    fmt = lambda v, p: (f'{v:+.2f}' + ('*' if p == p and p < 0.05 else ' ')
                        + (f'(p={p:.2f})' if p == p else '(p= — )'))
    sig_rows = []
    for (ds, model), g in df.groupby(['dataset', 'model']):
        print(f'\n===== {ds} / {model} =====')
        header = (f"{'budget':>6} {'baseline':>14} {'Δcount':>16} {'Δmlp':>16} "
                  f"{'Δoracle†':>16} {'Δshuffle':>16} {'count_dist':>10} "
                  f"{'mlp_dist':>9} {'n_edges':>8}")
        print(header)
        for budget, gb in sorted(g.groupby('budget')):
            base = gb[gb.reg_type == 'none'].drop_duplicates(
                ['seed', 'run'], keep='last').set_index(['seed', 'run'])
            if base.empty:
                print(f'{budget:>6}   (no baseline yet)')
                continue
            cells, covs = {}, {'count_dist': np.nan, 'mlp_dist': np.nan, 'n_edges': np.nan}
            for (rt, tr), cond in gb[gb.reg_type != 'none'].groupby(
                    ['reg_type', 'penalty_transform']):
                cond = cond.drop_duplicates(['lambda', 'seed', 'run'], keep='last')
                lam = cond.groupby('lambda')['best_valid'].mean().idxmax()
                sel = cond[cond['lambda'] == lam].drop_duplicates(
                    ['seed', 'run'], keep='last').set_index(['seed', 'run'])
                common = sel.index.intersection(base.index)
                if len(common) == 0:
                    continue
                d = (sel.loc[common, 'test_at_best_valid']
                     - base.loc[common, 'test_at_best_valid']).values
                key = rt if tr in ('', 'none') else f'{rt}/{tr}'
                cells[key] = (d.mean(), paired_pvalue(d), lam)
                dist = pd.to_numeric(sel.get('cooc_oracle_dist'), errors='coerce').mean()
                if rt == 'count':
                    covs['count_dist'] = dist
                    covs['n_edges'] = pd.to_numeric(sel.get('n_prior_edges'),
                                                    errors='coerce').mean()
                elif rt == 'mlp' and tr in ('', 'none'):
                    covs['mlp_dist'] = dist
                if key != 'oracle':
                    sig_rows.append((ds, model, budget, key, d.mean(),
                                     cells[key][1], lam))
            b = base['test_at_best_valid']
            row = f'{int(budget):>6} {b.mean():>8.2f}±{b.std():<5.2f}'
            for key in ('count', 'mlp', 'oracle', 'mlp/shuffle'):
                row += f" {fmt(*cells[key][:2]) if key in cells else '        —      ':>16}"
            row += (f" {covs['count_dist']:>10.3f}" if covs['count_dist'] == covs['count_dist'] else f" {'—':>10}")
            row += (f" {covs['mlp_dist']:>9.3f}" if covs['mlp_dist'] == covs['mlp_dist'] else f" {'—':>9}")
            row += (f" {covs['n_edges']:>8.0f}" if covs['n_edges'] == covs['n_edges'] else f" {'—':>8}")
            print(row)
        print('† oracle uses all labels incl. test — leaky diagnostic upper bound.')

    sig = [r for r in sig_rows if r[5] == r[5] and r[5] < 0.05]
    print('\n===== significant (p<0.05) leakage-free effects, largest first =====')
    for ds, model, budget, key, delta, p, lam in sorted(sig, key=lambda r: -r[4]):
        print(f'  {ds:<17} {model:<12} budget={int(budget):<3} {key:<12} '
              f'Δ={delta:+.2f}  p={p:.3f}  λ*={lam}')
    if not sig:
        print('  (none yet)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_dir', type=str, nargs='+', default=['results/lowlabel'],
                    help='each dir is analyzed separately — never pooled')
    args = ap.parse_args()
    for d in args.result_dir:
        analyze_dir(d)


if __name__ == '__main__':
    main()
