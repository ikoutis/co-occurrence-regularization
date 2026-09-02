"""
E4 (transfer experiment): build a co-occurrence prior from a LEGITIMATE
source — the labeled past of a temporal graph — and save it for --cooc_file.

For ogbn-arxiv with --year_split Y: the source pool is every paper with
node_year < Y (fully labeled: it is the past), the prior is the class-pair
count matrix on source-source edges (Laplace-smoothed, row-normalized —
the same estimator as --count_reg, applied to the past instead of the
training split). Nothing from the target pool (year >= Y), where all
training/validation/test nodes live, is touched.

Also reports how far this prior sits from (a) the all-label matrix of the
whole graph and (b) the all-label matrix of the target-only induced
subgraph — i.e. the temporal drift a transferred prior has to survive.

Usage (from medium_graph/):
    python make_cooc_prior.py --dataset ogbn-arxiv --year_split 2018 \
        --out priors/arxiv_lt2018_all.pt
    python make_cooc_prior.py --dataset ogbn-arxiv --year_split 2018 \
        --source_per_class 20 --out priors/arxiv_lt2018_m20.pt   # scarce-source variant
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops

from dataset import load_dataset
from regularization import estimate_cooccurrence_matrix, count_cooccurrence_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ogbn-arxiv')
    ap.add_argument('--data_dir', type=str, default='./data/')
    ap.add_argument('--year_split', type=int, required=True,
                    help='source pool = node_year < this; target pool = the rest')
    ap.add_argument('--source_per_class', type=int, default=0,
                    help='if >0, estimate the prior from only this many (seeded) '
                         'source labels per class — how much past is needed?')
    ap.add_argument('--smoothing', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    dataset = load_dataset(args.data_dir, args.dataset)
    if 'node_year' not in dataset.graph:
        raise SystemExit(f'{args.dataset} carries no node_year')
    label = dataset.label.reshape(-1).long()
    years = torch.as_tensor(dataset.graph['node_year']).reshape(-1)
    n = dataset.graph['num_nodes']
    c = int(label.max().item()) + 1

    edge_index = to_undirected(dataset.graph['edge_index'])
    edge_index, _ = remove_self_loops(edge_index)
    onehot = F.one_hot(label, c).float()

    source_mask = years < args.year_split
    target_mask = ~source_mask
    print(f'{args.dataset}: {n} nodes, {edge_index.shape[1]} directed edges, {c} classes')
    print(f'source (<{args.year_split}): {int(source_mask.sum())} nodes | '
          f'target (>={args.year_split}): {int(target_mask.sum())} nodes')

    if args.source_per_class > 0:
        torch.manual_seed(args.seed)
        keep = torch.zeros(n, dtype=torch.bool)
        src_idx = torch.where(source_mask)[0]
        for cls in label[src_idx].unique():
            idx_c = src_idx[label[src_idx] == cls]
            keep[idx_c[torch.randperm(idx_c.numel())[:args.source_per_class]]] = True
        source_mask = keep
        print(f'subsampled source labels: {int(source_mask.sum())} nodes '
              f'({args.source_per_class}/class)')

    # class coverage of the source pool (classes absent from the past get the
    # smoothing-only uniform row — worth knowing)
    counts = torch.bincount(label[source_mask], minlength=c)
    print(f'source labels per class: min {int(counts.min())}, median '
          f'{int(counts.float().median())}, max {int(counts.max())}; '
          f'{int((counts == 0).sum())} classes with no source labels')

    C_src, n_edges = count_cooccurrence_matrix(onehot, edge_index, source_mask, args.smoothing)
    C_full = estimate_cooccurrence_matrix(onehot, edge_index, c, edge_index.device)
    C_tgt, n_tgt_edges = count_cooccurrence_matrix(onehot, edge_index, target_mask, 0.0)
    rel = lambda A, B: ((A - B).norm() / B.norm().clamp(min=1e-12)).item()
    print(f'prior estimated from {n_edges} source-source edges')
    print(f'rel. Frobenius dist to all-label matrix of the WHOLE graph : {rel(C_src, C_full):.4f}')
    print(f'rel. Frobenius dist to all-label matrix of the TARGET graph: {rel(C_src, C_tgt):.4f} '
          f'({n_tgt_edges} target-target edges)')
    print(f'diagonal mass (homophily) — prior {C_src.diagonal().mean():.3f} | '
          f'target {C_tgt.diagonal().mean():.3f}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    obj = {'C': C_src.cpu(),
           'meta': {'dataset': args.dataset, 'year_split': args.year_split,
                    'source_per_class': args.source_per_class, 'smoothing': args.smoothing,
                    'n_edges': n_edges, 'n_source_nodes': int(source_mask.sum()),
                    'dist_to_full_oracle': rel(C_src, C_full),
                    'dist_to_target_oracle': rel(C_src, C_tgt)}}
    tmp = f'{args.out}.tmp.{os.getpid()}'
    torch.save(obj, tmp)
    os.replace(tmp, args.out)          # atomic: concurrent SLURM tasks may race here
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
