# Tuned GT Baselines: Published Configs, Fixes, and Gap Search

Goal (EXPERIMENT_PLAN.md Phase 1 / E1.1): GT baselines that reproduce the
numbers published by the Polynormer and SGFormer papers, so that any
regularization gain is measured against what those papers optimized —
not against an undertuned stand-in.

## Two implementation bugs found and fixed

### 1. Polynormer's global attention was never trained (critical)

The official Polynormer trains in two phases: `local_epochs` with
`model._global = False` (local attention only), then flips
`model._global = True` and trains `global_epochs` more, warm-starting from
the best-validation local checkpoint. The local `polynormer.py` faithfully
implements the architecture — including the `_global` switch — but
`main.py` never set it. **Every Polynormer result produced by this codebase
so far (including all Polynormer rows in FINDINGS.md Tables 5–6) trained
and evaluated only the local GATConv branch with the local prediction head;
the global attention module never received a gradient.**

Two consequences:
- The Polynormer baselines were not Polynormer. The 70.16 amazon-computer
  baseline vs 94.07 published is mostly explained by this plus the generic
  hyperparameters.
- The interpretation in FINDINGS §6 ("GTs without local MP gain the most")
  is inverted for these rows: the model actually tested had *only* local
  message passing (GAT-style attention over edges) and *no* global
  component.

Fix: `--local_epochs` / `--global_epochs` implement the official schedule
(requires `--save_model` for the phase-switch checkpoint reload). When
neither is set, the legacy single-phase behavior is kept but main.py now
prints a loud warning.

### 2. SGFormer's transformer depth was tied to its GCN depth

Official SGFormer has `--num_layers` (GCN backbone) and `--ours_layers`
(transformer branch, = 1 in every published config). The local
`parse_method` passed the single `--layers` value to **both**, so e.g.
`--layers 3` built a 3-layer transformer branch — a configuration the paper
never used. The official two-group optimizer (separate weight decay for the
transformer branch) was also missing.

Fix: `--tr_layers` (default 1), `--tr_dropout`, `--tr_weight_decay`
(enables the two-group optimizer). **Breaking change:** `--layers` now
controls only the GCN backbone; old sgformer scripts that relied on
`--layers` setting the transformer depth behave differently.

## Published config coverage

| | Polynormer | SGFormer |
|---|---|---|
| Published config available | roman-empire, amazon-ratings, minesweeper, questions, amazon-computer, amazon-photo, coauthor-cs, coauthor-physics, wikics → `polynormer_tuned.sh` | cora, citeseer, pubmed, chameleon → `sgformer_tuned.sh` |
| No published config → search | cora, citeseer, pubmed, chameleon, squirrel | amazon-computer, amazon-photo, coauthor-cs, coauthor-physics, wikics, roman-empire, amazon-ratings, minesweeper, questions, squirrel |

Notes:
- SGFormer/squirrel: the official config uses `--method difformer`
  (a different attention mechanism) — not reproducible here; searched
  instead.
- SGFormer chameleon/squirrel published numbers are on the **filtered**
  datasets (Platonov et al.); this repo loads the old versions, so
  published values are loose references until the loaders are switched.
- Every dataset has a published config for at least one GT, which anchors
  the search spaces for the other.

## Acceptance targets

Polynormer (official repo README, single-run):
roman-empire 92.48 · amazon-ratings 55.04 · minesweeper 97.19 (ROC) ·
questions 78.35 (ROC) · amazon-computer 94.07 · amazon-photo 96.67 ·
coauthor-cs 95.28 · coauthor-physics 97.14 · wikics 81.20

SGFormer (paper, 20-per-class splits):
cora 84.5 · citeseer 72.6 · pubmed 80.3

**Gate:** a (model, dataset) baseline is usable for regularization
experiments only once the 10-run mean is within ~1 point of its target
(where a target exists). Anything further off means a remaining
implementation/protocol difference — investigate before burning compute
on sweeps.

## Split-protocol warning

The previous GT sweep (`submit_sweep_mlp_gt_full.sbatch`) applied
`--rand_split_class` to amazon-computer/photo and coauthor-cs/physics,
while the GNN sweep (`run_gnn.sh`) used the loader's fixed splits for those
datasets. The GT-vs-GNN Δ comparisons in FINDINGS §6 therefore mixed split
protocols on those four datasets. The new scripts mirror `run_gnn.sh`
exactly.

## Workflow

```bash
cd GTs_baselines

# 1. Verify published configs reproduce published numbers (13 tasks)
sbatch submit_gt_tuned.sbatch          # -> results_tuned/

# 2. In parallel: search the gaps (15 tasks, 3 runs/config)
sbatch submit_gt_search.sbatch         # -> gt_search/results/
python analyze_gt_search.py            # validation-selected winners

# 3. Confirm each search winner with 10 runs, then treat as baseline.

# 4. Only after the gates pass: re-run the regularization sweep (E2.2)
#    and the GT placebo, using these configs + --paired_seeds.
```
