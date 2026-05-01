"""
Co-occurrence recovery experiment.

Question: given a graph and the oracle co-occurrence matrix C* (computed from
ALL true labels), can a model find a soft labeling that matches C* using only
the co-occurrence loss — no cross-entropy at all?

Setup:
  - Generate a Stochastic Block Model graph with known label structure
  - Compute C* from all true labels
  - Train GCN / GT / MLP with loss = edge_loss(softmax(out), edge_index, penalty)
  - Evaluate: Frobenius error ||C_pred - C*||_F and clustering accuracy

Three model families:
  GCN  — uses graph topology via message passing
  GT   — pure all-pairs self-attention, no positional encoding, edge_index ignored
  MLP  — no graph structure at all

The key hypothesis: GTs cannot recover co-occurrence statistics on their own
(they ignore topology), which is precisely why the explicit penalty helps them
more than it helps GNNs in the main paper experiments.

Usage:
  python cooc_recovery.py [--n_per_class 500] [--c 5] [--epochs 1000] [--runs 5]
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Co-occurrence helpers
# ---------------------------------------------------------------------------

def estimate_cooccurrence_matrix(probs, edge_index, c):
    src, dst = edge_index
    co = torch.matmul(probs[src].t(), probs[dst])
    co = (co + co.t()) / 2.0
    row_sum = co.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return co / row_sum


def edge_loss(probs, edge_index, penalty_matrix):
    src, dst = edge_index
    p_src = probs[src]
    p_dst = probs[dst]
    pen = torch.matmul(p_src, penalty_matrix)
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
    x = torch.randn(n, 16)   # pure noise features — no label information
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


class GT(nn.Module):
    """Pure all-pairs self-attention transformer.  edge_index is ignored —
    every node attends to every other node.  No positional encoding.
    This is the minimal GT that has no inductive topology bias."""

    def __init__(self, in_dim, hidden, out_dim, layers=3, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_proj = nn.Linear(hidden, out_dim)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index=None):
        # x: (N, in_dim) — treat all nodes as one sequence
        x = self.input_proj(x).unsqueeze(0)   # (1, N, hidden)
        x = self.transformer(x).squeeze(0)     # (N, hidden)
        return self.output_proj(x)


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
    for _ in range(epochs):
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

    pred_co = estimate_cooccurrence_matrix(probs, edge_index, c).cpu()
    fro_err = (pred_co - true_co).norm(p='fro').item()
    acc = clustering_accuracy(pred_labels, true_labels, c)

    return fro_err, acc


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
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available()
                          else 'cpu')
    print(f"Device: {device}")
    print(f"n={args.n_per_class * args.c} ({args.n_per_class}/class), "
          f"c={args.c}, epochs={args.epochs}, runs={args.runs}")

    conditions = {
        'homo':   dict(p_intra=0.10, p_inter=0.01),
        'hetero': dict(p_intra=0.02, p_inter=0.08),
    }

    models = [
        ('GCN', lambda: GCN(16, args.hidden, args.c, args.layers)),
        ('GT',  lambda: GT(16, args.hidden, args.c, args.layers, args.heads)),
        ('MLP', lambda: MLP(16, args.hidden, args.c, args.layers)),
    ]

    for regime, sbm_params in conditions.items():
        print(f"\n{'='*60}")
        print(f"Regime: {regime}  "
              f"(p_intra={sbm_params['p_intra']}, p_inter={sbm_params['p_inter']})")
        print('='*60)

        x, edge_index, true_labels = generate_sbm(
            args.n_per_class, args.c, seed=42, **sbm_params
        )
        c = args.c

        true_probs = F.one_hot(true_labels, c).float()
        true_co = estimate_cooccurrence_matrix(true_probs, edge_index, c)
        penalty_matrix = -torch.log(true_co + 1e-6)

        print(f"Oracle C* (row-normalized):")
        print(np.round(true_co.numpy(), 3))

        for model_name, build in models:
            model = build()
            fro_errs, accs = [], []

            for run in range(args.runs):
                torch.manual_seed(run)
                np.random.seed(run)
                fro_err, acc = run_once(
                    model, x, edge_index, penalty_matrix, true_labels, c,
                    args.epochs, args.lr, device
                )
                fro_errs.append(fro_err)
                accs.append(acc)

            print(f"  {model_name:4s} | "
                  f"Fro error: {np.mean(fro_errs):.4f} ± {np.std(fro_errs):.4f} | "
                  f"Cluster acc: {np.mean(accs)*100:.1f} ± {np.std(accs)*100:.1f}%")


if __name__ == '__main__':
    main()
