"""
Analysis for the transfer experiment (E4) — see TRANSFER_README.md.

Question: does a co-occurrence prior counted on a LEGITIMATE source (the
labeled past of ogbn-arxiv) help a model trained on a small label budget of
the future — and how does that compare with (a) estimating the prior from
the budget itself (count / mlp), (b) the leaky all-label oracle, (c) simply
adding the past's labels to training?

Reads results/transfer_<mode>[_<prior>] dirs. Within each dir, deltas are
paired (seed, run) against that dir's own baseline; the cross-mode
"labels-vs-prior" comparison is paired too — keep/labels share the exact
same target sample per run (same seed), drop's sample is the same nodes
relabeled.

Usage:  python analyze_transfer.py [--root results]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from analyze_lowlabel import load, paired_pvalue


def select(cond, base):
    """validation-selected lambda, paired deltas vs base (indexed by seed,run)"""
    cond = cond.drop_duplicates(['lambda', 'seed', 'run'], keep='last')
    lam = cond.groupby('lambda')['best_valid'].mean().idxmax()
    sel = cond[cond['lambda'] == lam].drop_duplicates(['seed', 'run'], keep='last').set_index(['seed', 'run'])
    common = sel.index.intersection(base.index)
    if len(common) == 0:
        return None
    d = (sel.loc[common, 'test_at_best_valid'] - base.loc[common, 'test_at_best_valid']).values
    dist = pd.to_numeric(sel.get('cooc_oracle_dist'), errors='coerce').mean()
    return d, lam, dist, len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results')
    args = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(args.root, 'transfer_*')) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f'no results/transfer_* dirs under {args.root}')

    frames = {}
    for d in dirs:
        try:
            frames[os.path.basename(d)] = load([d])
        except SystemExit:
            pass
    fmt = lambda d, p: f'{d:+.2f}' + ('*' if p == p and p < 0.05 else ' ') + (f'(p={p:.2f})' if p == p else '(p= — )')

    def baseline_of(mode):
        key = f'transfer_{mode}'
        if key not in frames:
            return None
        df = frames[key]
        return df[df.reg_type == 'none']

    budgets = sorted({b for df in frames.values() for b in df['budget'].unique()})
    for budget in budgets:
        print(f'\n===================== budget {int(budget)}/class =====================')
        bases = {}
        for mode in ('keep', 'drop', 'labels'):
            b = baseline_of(mode)
            if b is None:
                continue
            b = b[b.budget == budget].drop_duplicates(['seed', 'run'], keep='last').set_index(['seed', 'run'])
            if not b.empty:
                bases[mode] = b
                print(f'  baseline[{mode:<6}] = {b.test_at_best_valid.mean():6.2f} ± {b.test_at_best_valid.std():.2f}  (n={len(b)})')
        if 'keep' in bases and 'labels' in bases:
            common = bases['keep'].index.intersection(bases['labels'].index)
            d = (bases['labels'].loc[common, 'test_at_best_valid'] - bases['keep'].loc[common, 'test_at_best_valid']).values
            print(f'  value of the past AS LABELS (labels − keep, paired): {fmt(d.mean(), paired_pvalue(d))}')
        print(f"\n  {'dir':<22} {'condition':<18} {'Δ vs own baseline':>18} {'λ*':>5} {'prior_dist':>10} {'n':>3}")
        for name, df in frames.items():
            mode = name.split('_')[1]
            if mode not in bases:
                continue
            g = df[(df.budget == budget) & (df.reg_type != 'none')]
            for (rt, tr), cond in g.groupby(['reg_type', 'penalty_transform']):
                r = select(cond, bases[mode])
                if r is None:
                    continue
                d, lam, dist, n = r
                label = rt if tr in ('', 'none') else f'{rt}/{tr}'
                if rt == 'oracle':
                    label += '†'
                print(f'  {name:<22} {label:<18} {fmt(d.mean(), paired_pvalue(d)):>18} {lam:>5} '
                      f"{(f'{dist:.3f}' if dist == dist else '—'):>10} {n:>3}")
        print('  † oracle uses all labels incl. test (and, in keep/labels mode, the past) — leaky diagnostic.')
        print('  prior_dist = rel. Frobenius distance of the penalty source to the all-label matrix of the graph as trained on.')


if __name__ == '__main__':
    main()
