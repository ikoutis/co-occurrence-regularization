# oracle_gnn: oracle (upper-bound) regularization

The penalty matrix is computed once, before training, from the **true labels**:

```
true_probs = one_hot(labels)                  # [N, C]
co_matrix  = estimate_cooccurrence(true_probs, edge_index, C)
penalty    = -log(co_matrix + 1e-6)
```

It is then frozen for the entire run. Same per-step normalization
(`scale = loss.detach() / (reg_loss.detach().abs() + 1e-8)`) as in `mlp_gnn`
and `gnn_gnn`.

## What this experiment is for

This is an **idealized upper-bound experiment**, not a practical method.
It uses test-set labels to build the penalty matrix, so it is not
deployable. The goal is to answer:

> If we had perfect knowledge of the class co-occurrence structure of
> the graph, how much would co-occurrence regularization help?

This bounds the headroom available to any estimation strategy
(MLP-based, GNN-based, or future approaches).

## Comparison to the other sweeps

| Sweep | Penalty source | When computed | Updated? |
|---|---|---|---|
| `mlp_gnn` | MLP's softmax outputs | After 500 epochs of MLP pre-training | No (frozen) |
| `gnn_gnn` | GNN's own softmax outputs | Every 50 epochs during training | Yes (dynamic) |
| **`oracle_gnn`** | **True labels** | **Once, before training** | **No (frozen)** |

## How to read the results

For each (dataset, GNN) cell, compare three numbers at the best lambda:

- `mlp_gnn` Δ: practical method, MLP-derived penalty
- `gnn_gnn` Δ: practical method, GNN-derived penalty
- **`oracle_gnn` Δ: theoretical upper bound**

Three patterns are possible:

1. **oracle ≫ mlp_gnn**: estimator quality is the bottleneck. Better
   estimation strategies are worth pursuing.
2. **oracle ≈ mlp_gnn**: MLP already captures most of the available
   signal. Further refinement has diminishing returns; effort is better
   spent elsewhere (architecture, training).
3. **oracle ≈ baseline**: the penalty structure itself has limited
   capacity to help on this dataset, regardless of source. Co-occurrence
   regularization is fundamentally a poor fit here.

## Coverage

All 14 datasets × 4 methods (MLP, gcn, sage, gat) × 5 lambdas
(0, 0.01, 0.1, 0.2, 0.4) = 56 SLURM tasks, 5 runs each.

Note that **MLP** is included with the full lambda sweep (unlike
`mlp_gnn` where MLP only had a baseline). This tests whether an MLP
that ignores edges in its forward pass can still benefit from a
graph-aware regularization signal.
