# mlp_gt: MLP-derived co-occurrence regularization for Graph Transformers

This folder contains the pilot experiment applying the `mlp_gnn` regularization
strategy to Graph Transformer models.

## What this is

Same idea as `medium_graph/results/mlp_gnn/`:
1. Pre-train a fresh MLP (500 epochs) on node features only
2. Compute co-occurrence matrix from MLP's softmax outputs
3. Freeze penalty matrix `= -log(co_matrix + 1e-6)`
4. Train a Graph Transformer with per-step normalized regularization:
   `loss += lambda * (task_loss / reg_loss) * reg_loss`

## Why this pilot dataset: cora

| Property | Value |
|---|---|
| Nodes | 2,708 |
| Classes | 7 |
| MLP accuracy | 55.72% |
| Best GNN (mlp_gnn) | GCN/GAT ~80.5% |
| Best mlp_gnn Δ | GAT +1.04, GCN +0.54 |
| GPU needed | a100_10g (fits easily) |

Cora is the smallest dataset with clear mlp_gnn wins and a large MLP-to-GNN gap,
making it the cleanest pilot for GTs.

## Models included

Only linear-attention GTs (memory-safe, scalable to larger datasets later):

| Task | Model | Architecture notes |
|---|---|---|
| 0 | SGFormer | GCN backbone + fast global attention; graph_weight=0.5 |
| 1 | Polynormer | Local + equivariant global attention; proven on citation graphs |
| 2 | NodeFormer | Kernel-approximated all-pair attention |

GPS, Exphormer, GOAT are excluded from this pilot (quadratic attention →
much higher memory / slower on larger graphs).

## Lambda sweep

{0 (baseline), 0.01, 0.1, 0.2, 0.4} — same as mlp_gnn for direct comparison.

## How to run

```bash
# From GTs_baselines/ directory
sbatch mlp_gt/submit_sweep_mlp_gt.sbatch
```

## Expected comparison

Results land in `mlp_gt/results/cora/{sgformer,polynormer,nodeformer}.csv`.
Compare directly to `medium_graph/results/mlp_gnn/cora/MPNN_{gcn,sage,gat}.csv`.

The key question: do GTs benefit more or less than GCN/SAGE/GAT from the
MLP-derived penalty on cora? GTs already capture long-range structure via
global attention; the local co-occurrence penalty may be redundant — or it
may complement the global attention by sharpening edge-level predictions.
