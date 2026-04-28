# mlp_gt full sweep: co-occurrence regularization for Graph Transformers

## Motivation

The cora pilot (polynormer, sgformer on cora) revealed a clean pattern:

| Model | Local aggregation | Baseline | Best λ | Δ |
|---|---|---|---|---|
| Polynormer | attention-only | 75.72 | 0.01 | **+2.20** |
| SGFormer | GCN backbone + global attn | 81.16 | — | **−1.16** |
| GCN | message passing | 80.48 | 0.4 | +0.54 |
| GAT | message passing | 80.28 | 0.4 | +1.04 |

The benefit of co-occurrence regularization is not "GTs vs GNNs" — it is
"does the model have local message passing or not?"

- **No local MP (Polynormer, NodeFormer):** global attention smears
  edge-level label coherence. The penalty matrix restores exactly that
  signal → large gains.
- **Local MP built in (SGFormer's GCN backbone, standard GCN/GAT/SAGE):**
  local label coherence is already encoded by message passing → penalty
  is redundant and potentially destabilizing.

This is a mechanistically testable claim: the gain from co-occurrence
regularization scales inversely with the strength of local message passing.

## Hypothesis

**H1 (main):** Polynormer and NodeFormer gain significantly more from
MLP-derived co-occurrence regularization than GCN/GAT/SAGE on the same
dataset, because they lack a local-MP inductive bias.

**H2 (control):** SGFormer, which has a GCN backbone, gains little or
nothing — matching GNN-level performance or declining — confirming that
the benefit is specifically tied to missing local aggregation.

**H3 (ceiling):** The oracle penalty experiment shows that GNN gains are
already ≤2% absolute on most datasets. If GT gains (H1) exceed this
ceiling, it further validates that GTs are the right target for this
regularization strategy.

## Models

Only linear-attention GTs (O(N) complexity, feasible on all 14 datasets):

| Model | Local aggregation mechanism | Expected response |
|---|---|---|
| Polynormer | Equivariant local attention (not MP) | Large gains |
| NodeFormer | Kernel-approx all-pair attention (no local) | Large gains |
| SGFormer | GCN backbone + simple global attention | Flat or negative |

Quadratic models (GPS, Exphormer, GOAT) excluded: memory/time prohibitive
on large graphs (roman-empire 22k, coauthor-physics 34k, amazon-ratings 24k).

## Datasets

All 14 datasets from the mlp_gnn GNN sweep, for direct comparison:

| # | Dataset | Nodes | Type | GNN best Δ (mlp_gnn) |
|---|---|---|---|---|
| 0 | cora | 2,708 | homophilic | GAT +1.04 |
| 1 | citeseer | 3,327 | homophilic | SAGE +0.92 |
| 2 | pubmed | 19,717 | homophilic | GCN +0.46 |
| 3 | amazon-ratings | 24,492 | heterophilic | SAGE +0.88 |
| 4 | minesweeper | 10,000 | heterophilic | SAGE +0.45 |
| 5 | questions | 11,452 | heterophilic | (degenerate GNN) |
| 6 | roman-empire | 22,662 | heterophilic | SAGE +0.36 |
| 7 | amazon-computer | 13,752 | homophilic | GAT +0.41 |
| 8 | amazon-photo | 7,650 | homophilic | SAGE +0.21 |
| 9 | coauthor-cs | 18,333 | homophilic | GCN +0.09 |
| 10 | coauthor-physics | 34,493 | homophilic | GAT +0.18 |
| 11 | wikics | 11,701 | homophilic | GCN +0.08 |
| 12 | chameleon | 2,277 | heterophilic | GCN +0.59 |
| 13 | squirrel | 5,201 | heterophilic | GCN +0.41 |

## Lambda sweep

`{0 (baseline), 0.01, 0.1, 0.2, 0.4}` — same as mlp_gnn for direct comparison.

Note: the cora pilot showed Polynormer peaking at λ=0.01 (smallest
non-zero value). If this pattern holds broadly, a finer grid around
{0.001, 0.005, 0.01, 0.02} would be a natural follow-up.

## Hyperparameters

Fixed defaults per model (tuned on cora pilot, applied uniformly across
all datasets). Not per-dataset tuned — noted as a limitation.

| Model | hidden | layers | heads | lr | dropout | other |
|---|---|---|---|---|---|---|
| Polynormer | 128 | local=2, global=2 | 4 | 0.01 | 0.3 | — |
| NodeFormer | 128 | global=2 | 4 | 0.01 | 0.3 | use_bn |
| SGFormer | 256 | 3 | 1 | 0.01 | 0.3 | use_graph, use_act, use_bn, use_residual, alpha=0.5, graph_weight=0.5 |

5 runs per configuration. 1000 epochs per run.

## Expected results layout

```
mlp_gt/results_full/
  cora/
    polynormer.csv
    nodeformer.csv
    sgformer.csv
  citeseer/
    ...
  ...
```

Compare directly to `medium_graph/results/mlp_gnn/` for GNN baselines.

## How to run

```bash
# From GTs_baselines/ directory
sbatch mlp_gt/submit_sweep_mlp_gt_full.sbatch
```

## What a positive result looks like for a paper

A NeurIPS-quality result requires:
1. Polynormer and NodeFormer gain ≥2× more than GCN/GAT/SAGE on the
   same datasets, consistently across ≥5 datasets.
2. SGFormer gains little or nothing (the control).
3. The differential correlates with the presence/absence of local MP.

This would establish co-occurrence regularization as a principled fix for
a known weakness of pure-attention GTs: the absence of local label-coherence
inductive bias that message passing provides for free.
