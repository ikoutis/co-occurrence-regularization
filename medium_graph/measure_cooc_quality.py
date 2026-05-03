"""
Q1 diagnostic: do regularized models produce predictions with co-occurrence
statistics closer to the oracle C*?

For each lambda condition (baseline + mlp_reg sweep), trains a GCN and measures:
  - Test accuracy (standard)
  - Normalized Frobenius distance to oracle: ||C_pred - C_oracle||_F / ||C_oracle||_F
    computed from ALL node predictions (not just test nodes)

If regularization works as claimed, fro_dist should decrease as lambda increases
(at least up to the optimal lambda). If fro_dist does not decrease, the accuracy
gains are due to something other than co-occurrence alignment.

Usage (from medium_graph/):
  python measure_cooc_quality.py --dataset cora --runs 3
  python measure_cooc_quality.py --dataset coauthor-cs --runs 3
  python measure_cooc_quality.py --dataset amazon-ratings --runs 3 --metric rocauc
"""

import argparse
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from parse import parse_method, parser_add_main_args
from logger import Logger
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from eval import evaluate
from regularization import estimate_cooccurrence_matrix, edge_loss
from model import MLP


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fro_dist(c_pred, c_oracle):
    """Normalized Frobenius distance."""
    return (c_pred - c_oracle).norm(p='fro').item() / c_oracle.norm(p='fro').item()


def train_and_measure(args, dataset, split_idx, c, d, device, penalty_matrix, run):
    """Train one GCN run and return (test_acc, fro_dist_to_oracle)."""
    fix_seed(run)

    model = parse_method(args, dataset.graph['num_nodes'], c, d, device)
    model.reset_parameters()
    criterion = nn.NLLLoss() if args.dataset not in ('questions',) else nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc if args.metric == 'rocauc' else eval_acc
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_idx = split_idx['train'].to(device)
    best_val, best_test = float('-inf'), float('-inf')

    # Oracle C* from all true labels (used only for measurement, not training)
    with torch.no_grad():
        true_probs = F.one_hot(dataset.label.squeeze(1), c).float().to(device)
        c_oracle = estimate_cooccurrence_matrix(true_probs, dataset.graph['edge_index'], c, device).cpu()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

        if args.dataset in ('questions',):
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1) \
                if dataset.label.shape[1] == 1 else dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].float())
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])

        if penalty_matrix is not None:
            probs = torch.sigmoid(out) if args.dataset in ('questions',) else torch.exp(out)
            reg = edge_loss(probs, dataset.graph['edge_index'], penalty_matrix)
            scale = loss.detach().abs() / reg.detach().abs().clamp(min=1e-4)
            loss = loss + args.lambda_val * scale * reg

        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]

    # Measure co-occurrence quality at end of training
    model.eval()
    with torch.no_grad():
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        probs = torch.sigmoid(out) if args.dataset in ('questions',) \
            else torch.softmax(out, dim=1)
        c_pred = estimate_cooccurrence_matrix(probs, dataset.graph['edge_index'], c, device).cpu()

    return best_test * 100, fro_dist(c_pred, c_oracle)


def main():
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.01, 0.1, 0.2, 0.4])
    args = parser.parse_args()
    args.gnn = 'gcn'
    args.use_reg = False
    args.mlp_reg = False
    args.oracle_reg = False
    args.save_model = False
    args.save_result = False

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Dataset: {args.dataset} | Runs: {args.runs}")

    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    if args.rand_split_class:
        split_idx = class_rand_splits(dataset.label, args.label_num_per_class,
                                      args.valid_num, args.test_num)
    else:
        split_idx = load_fixed_splits(args.data_dir, dataset, name=args.dataset)
        if isinstance(split_idx, list):
            split_idx = split_idx[0]

    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    print(f"n={n}, c={c}, d={d}, train={len(split_idx['train'])}")

    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    criterion = nn.NLLLoss() if args.dataset not in ('questions',) else nn.BCEWithLogitsLoss()

    # Pre-train MLP once (shared across lambda conditions)
    print(f"\nPre-training MLP for {args.mlp_epochs} epochs...")
    mlp = MLP(d, args.hidden_channels, c,
              num_layers=max(2, args.local_layers), dropout=args.dropout).to(device)
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_idx = split_idx['train'].to(device)
    for ep in range(args.mlp_epochs):
        mlp.train()
        mlp_opt.zero_grad()
        out = mlp(dataset.graph['node_feat'])
        out_ls = F.log_softmax(out, dim=1)
        loss = criterion(out_ls[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        mlp_opt.step()
    mlp.eval()
    with torch.no_grad():
        mlp_probs = torch.softmax(mlp(dataset.graph['node_feat']), dim=1)
        mlp_co = estimate_cooccurrence_matrix(mlp_probs, dataset.graph['edge_index'], c, device)
        mlp_penalty = -torch.log(mlp_co + 1e-6)
    print("MLP pre-training done.")

    # Results storage
    out_dir = f'results/cooc_quality'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/{args.dataset}_gcn.csv'

    conditions = [(None, 0.0)] + [(mlp_penalty, lam) for lam in args.lambdas]
    rows = []

    for penalty_matrix, lam in conditions:
        args.lambda_val = lam
        accs, fros = [], []
        label = "baseline" if penalty_matrix is None else f"mlp_reg λ={lam}"
        print(f"\n--- {label} ---")

        for run in range(args.runs):
            acc, fro = train_and_measure(args, dataset, split_idx, c, d,
                                         device, penalty_matrix, run)
            accs.append(acc)
            fros.append(fro)
            print(f"  run {run+1}: acc={acc:.2f}  fro={fro:.4f}")

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        mean_fro = np.mean(fros)
        std_fro  = np.std(fros)
        print(f"  → acc: {mean_acc:.2f} ± {std_acc:.2f}  |  fro: {mean_fro:.4f} ± {std_fro:.4f}")
        rows.append((label, mean_acc, std_acc, mean_fro, std_fro))

    # Save CSV
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'acc_mean', 'acc_std', 'fro_mean', 'fro_std'])
        writer.writerows(rows)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print(f"\n{'Condition':<20} {'Acc':>8} {'±':>6} {'Fro':>8} {'±':>6}")
    print('-' * 52)
    baseline_fro = rows[0][3]
    for label, ma, sa, mf, sf in rows:
        delta_fro = mf - baseline_fro
        marker = ' ↓' if delta_fro < -0.001 else (' ↑' if delta_fro > 0.001 else '')
        print(f"{label:<20} {ma:>8.2f} {sa:>6.2f} {mf:>8.4f} {sf:>6.4f}{marker}")


if __name__ == '__main__':
    main()
