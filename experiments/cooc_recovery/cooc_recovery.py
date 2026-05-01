"""
Co-occurrence recovery experiment.

Question: given a graph and the oracle co-occurrence matrix C* (computed from
ALL true labels), can a GNN find a soft labeling that matches C* using only
the co-occurrence loss — no cross-entropy at all?

Setup:
  - Generate a Stochastic Block Model graph with known label structure
  - Compute C* from all true labels
  - Train GNN / MLP with loss = edge_loss(softmax(out), edge_index, penalty)
  - Evaluate: Frobenius error ||C_pred - C*||_F and clustering accuracy

The experiment runs four conditions:
  gcn-homo, gcn-hetero, mlp-homo, mlp-hetero

Usage:
  python cooc_recovery.py [--n_per_class 500] [--c 5] [--epochs 1000] [--runs 5]
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Re-implement co-occurrence helpers locally (no sys.path hacks)
# ---------------------------------------------------------------------------

def estimate_cooccurrence_matrix(probs, edge_index, c):
    src, dst = edge_index
    co = torch.matmul(probs[src].t(), probs[dst])
    co = (co + co.t()) / 2.0
    row_sum = co.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return co / row_sum


def edge_loss(probs, edge_index, penalty_matrix):
    src, dst = edge_index
    p_src = probs[src]   # (E, C)
    p_dst = probs[dst]   # (E, C)
    # For each edge: sum_{i,j} p_src[i] * p_dst[j] * penalty[i,j]
    pen = torch.matmul(p_src, penalty_matrix)  # (E, C)
    return (pen * p_dst).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Graph generation: Stochastic Block Model
# ---------------------------------------------------------------------------

def generate_sbm(n_per_class, c, p_intra, p_inter, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_class * c
    labels = torch.repeat_interleave(torch.arange(c), n_per_class)

    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            p = p_intra if labels[i] == labels[j] else p_inter
            if rng.random() < p:
                rows += [i, j]
                cols += [j, i]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    # Node features: Gaussian noise (no label information)
    x = torch.randn(n, 16)
    return x, edge_index, labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=3):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index=None):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Clustering accuracy via Hungarian matching
# ---------------------------------------------------------------------------

def clustering_accuracy(pred_labels, true_labels, c):
    """Accuracy after optimal label permutation (Hungarian algorithm)."""
    confusion = np.zeros((c, c), dtype=np.int64)
    for t, p in zip(true_labels.numpy(), pred_labels.numpy()):
        confusion[t, p] += 1
    row, col = linear_sum_assignment(-confusion)
    return confusion[row, col].sum() / len(true_labels)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_once(model, x, edge_index, penalty_matrix, true_labels, c,
             epochs, lr, device):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = x.to(device)
    edge_index = edge_index.to(device)
    penalty_matrix = penalty_matrix.to(device)
    true_co = estimate_cooccurrence_matrix(
        F.one_hot(true_labels, c).float().to(device), edge_index, c
    ).cpu()

    model.to(device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        probs = torch.softmax(out, dim=1)
        loss = edge_loss(probs, edge_index, penalty_matrix)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        probs = torch.softmax(out, dim=1)
        pred_labels = probs.argmax(dim=1).cpu()

    # Co-occurrence Frobenius error
    pred_co = estimate_cooccurrence_matrix(probs, edge_index, c).cpu()
    fro_err = (pred_co - true_co).norm(p='fro').item()

    # Clustering accuracy (Hungarian)
    acc = clustering_accuracy(pred_labels, true_labels, c)

    return fro_err, acc, pred_co


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_per_class', type=int, default=500)
    parser.add_argument('--c', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available()
                          else 'cpu')
    print(f"Device: {device}")
    print(f"n={args.n_per_class * args.c} ({args.n_per_class}/class), "
          f"c={args.c}, epochs={args.epochs}, runs={args.runs}")

    # Two graph regimes
    conditions = {
        'homo':   dict(p_intra=0.10, p_inter=0.01),   # clear communities
        'hetero': dict(p_intra=0.02, p_inter=0.08),   # cross-class edges dominate
    }

    for regime, sbm_params in conditions.items():
        print(f"\n{'='*60}")
        print(f"Regime: {regime}  "
              f"(p_intra={sbm_params['p_intra']}, p_inter={sbm_params['p_inter']})")
        print('='*60)

        x, edge_index, true_labels = generate_sbm(
            args.n_per_class, args.c, seed=42, **sbm_params
        )
        c = args.c

        # Oracle co-occurrence matrix and penalty
        true_probs = F.one_hot(true_labels, c).float()
        true_co = estimate_cooccurrence_matrix(true_probs, edge_index, c)
        penalty_matrix = -torch.log(true_co + 1e-6)

        print(f"Oracle C* (row-normalized):")
        print(np.round(true_co.numpy(), 3))

        for model_name, ModelClass in [('GCN', GCN), ('MLP', MLP)]:
            model = ModelClass(x.shape[1], args.hidden, c, args.layers)
            fro_errs, accs = [], []

            for run in range(args.runs):
                torch.manual_seed(run)
                np.random.seed(run)
                fro_err, acc, _ = run_once(
                    model, x, edge_index, penalty_matrix, true_labels, c,
                    args.epochs, args.lr, device
                )
                fro_errs.append(fro_err)
                accs.append(acc)

            mean_fro = np.mean(fro_errs)
            std_fro  = np.std(fro_errs)
            mean_acc = np.mean(accs) * 100
            std_acc  = np.std(accs) * 100
            print(f"\n  {model_name:4s} | "
                  f"Fro error: {mean_fro:.4f} ± {std_fro:.4f} | "
                  f"Cluster acc: {mean_acc:.1f} ± {std_acc:.1f}%")


if __name__ == '__main__':
    main()
