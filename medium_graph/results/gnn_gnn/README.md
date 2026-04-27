# gnn_gnn: dynamic GNN-based regularization

The penalty matrix is rebuilt every `reg_update_freq=50` epochs from the
GNN's own predictions, starting at `reg_start_epoch=10`. Same per-step
normalization as `mlp_gnn` (`scale = loss.detach() / (reg_loss.detach().abs() + 1e-8)`).

## Why this subset of datasets

This sweep is a targeted comparison against `results/mlp_gnn/`, not a
full re-sweep. We selected 5 datasets where the dynamic approach has a
plausible chance of beating the static MLP-based penalty.

| Dataset | Why included |
|---|---|
| amazon-ratings | Cleanest `mlp_gnn` win (sage +0.91, monotonic in lambda). Stable baseline → good signal-to-noise to detect whether dynamic improves on it. |
| roman-empire | gcn was monotonic in `mlp_gnn` (+0.79), and SAGE has a huge MLP-to-GNN gap (66 → 87). Dynamic should produce a higher-quality penalty than MLP since the GNN has much better predictions. |
| minesweeper | Largest MLP-to-GNN gap in the sweep (MLP=51.4, sage=92.4). The MLP-derived penalty is the noisiest of any dataset; this is where the dynamic approach has the most theoretical room to improve. |
| cora | Strongest `mlp_gnn` win on a homophilic graph (gat +1.04). GAT's own predictions are reliable, so dynamic penalty should carry real signal. |
| citeseer | Largest individual win in `mlp_gnn` (gat +1.26). Same logic as cora. |

## Why other datasets were excluded

- **squirrel, chameleon**: GNN barely beats MLP (e.g., squirrel sage=37.6 vs MLP=40.6). The GNN's own co-occurrence estimates are not meaningfully better than MLP's, so dynamic offers little upside. Also: small graphs with high std → noise dominates.
- **coauthor-cs, coauthor-physics, questions**: at ceiling. Already-saturated metrics leave no room for any regularization to help, regardless of penalty source.
- **amazon-computer, amazon-photo, wikics, pubmed**: `mlp_gnn` already wash on these; not informative for comparing penalty sources.

## Hypothesis

If dynamic > static, we should see the largest improvements on
**minesweeper** and **roman-empire/sage** — both cases where the MLP
penalty is poor-quality (low MLP accuracy or large MLP-GNN gap), and
the GNN itself produces a much better signal that dynamic can leverage.

If dynamic ≈ static, the static MLP approach is sufficient and we
prefer it for simplicity (no warm-up, no oscillation, fully decoupled).

If dynamic < static on heterophilic, the dynamic penalty is reinforcing
the GNN's own mistakes (circular reasoning) — a known failure mode
for self-distillation on noisy outputs.
