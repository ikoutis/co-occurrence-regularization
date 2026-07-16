# Placebo Experiment: Is It the Information, or Just Regularization?

## Question

The mlp_gnn gains could come from two very different sources:

- **(A) Co-occurrence information** — the penalty matrix tells the model
  *which* class pairs belong on edges.
- **(B) Generic regularization** — any penalty of this bilinear form
  smooths/sharpens predictions, and the gain has nothing to do with the
  matrix's content.

If (B), the method is a complicated way to do label smoothing and the paper
narrative collapses. This experiment separates them.

## The concern with a naive placebo

A permuted penalty matrix is **not a clean placebo when co-occurrence
statistics are similar across class pairs**: permuting a near-uniform
matrix returns a near-identical matrix, so "the placebo also worked" is
indistinguishable from "the placebo was the same matrix." Two further
subtleties:

1. **Class-relabeling permutations don't work at all.** A simultaneous
   row/column permutation maps the diagonal to the diagonal, so it
   *preserves homophily* — the dominant signal on homophilic graphs. A
   placebo that keeps the main signal intact tests nothing.
2. On homophilic datasets the off-diagonal entries are all small and
   similar, so even entry-wise shuffling restricted to off-diagonals is
   weak there.

## Design: two controls + a power diagnostic

Three penalty conditions, all built from the same MLP co-occurrence matrix
`C`, all swept over λ ∈ {0.05, 0.1, 0.2, 0.4}:

| Condition (`--penalty_transform`) | What it does | What it tests |
|---|---|---|
| `none` | true `-log(C + ε)` | the method |
| `shuffle` | permute **all** entries of `C` (incl. diagonal), re-row-normalize | destroys all class semantics incl. homophily. If this matches `none`, gains are generic regularization (B). |
| `homophily` | keep the diagonal of `C`, spread each row's remaining mass uniformly off-diagonal | keeps the homophily level, removes class-pair structure. Separates "match your neighbors" from "match the *right class pairs*". |

**Power diagnostic (answers the "not so clean" concern):** every run logs

- `penalty_dist` = ‖P_transformed − P_true‖_F / ‖P_true‖_F — how much the
  transform actually changed the penalty. **A shuffle row with
  `penalty_dist` ≈ 0 has no statistical power and must not be
  interpreted**, exactly the degenerate case where co-occurrence
  statistics are similar across pairs. The analysis prints this column
  next to every Δ so uninterpretable rows are visible, not silent.
- `offdiag_cv` = coefficient of variation of the off-diagonal penalties —
  how much class-pair structure exists to destroy in the first place.

The dataset selection also works in our favor: the placebo has the most
power exactly where the method's most interesting wins are — heterophilic
datasets (amazon-ratings, squirrel, roman-empire) have structured
off-diagonal co-occurrence (high `offdiag_cv`), while on homophilic
datasets the `homophily` control carries the argument (there, the honest
expected outcome is that the homophily control captures much of the gain,
and the claim becomes correspondingly narrower).

## Interpretation matrix

| Outcome at λ* (validation-selected) | Reading |
|---|---|
| `none` > `shuffle` ≈ baseline | gains are information-driven ✓ |
| `none` ≈ `shuffle` > baseline (with `penalty_dist` high) | gains are generic regularization ✗ |
| `none` ≈ `homophily` > `shuffle` | gains come from homophily level only — method reduces to a homophily prior |
| `none` > `homophily` > `shuffle` | class-pair structure adds value beyond homophily — strongest possible result |

## Protocol

- **Paired runs** (`--paired_seeds`): run *k* of every condition uses the
  same split, the same model init, and the same training-noise stream
  (re-seeded after MLP pre-training, which otherwise desynchronizes the
  RNG). Differences between conditions are attributable to the loss term
  alone; enables Wilcoxon signed-rank tests on per-run deltas.
- **10 runs**, λ selected by **mean validation accuracy** (test never used
  for selection). Analysis: `analyze_placebo.py`.
- The shuffle permutation is fixed within a run and varies across runs
  (seeded by `seed + 5000 + run`), so the result is not an artifact of one
  unlucky permutation.
- Hyperparameters: tunedGNN-optimized configs from `run_gnn.sh`, verbatim.
  The goal is beating the optimized baselines from the tunedGNN paper, so
  those configs are the fixed reference point.

## Targets (8 sbatch array tasks)

The pairs with the largest reported mlp_gnn wins in FINDINGS.md Table 2:
cora/GCN, cora/GAT, citeseer/GAT, squirrel/GAT, squirrel/GCN,
amazon-ratings/SAGE, amazon-ratings/GCN, roman-empire/GAT.

roman-empire/GCN is excluded: its baseline (28.54 vs ~91 published) is
degenerate and nothing measured on top of it is interpretable.

## Running

```bash
cd medium_graph
sbatch submit_placebo.sbatch          # single submission, array 0-7
# after completion:
python analyze_placebo.py --result_dir results/placebo
```

Output: `results/placebo/placebo_summary.md` — one row per
(dataset, model, condition) with λ*, paired Δ, Wilcoxon p, and the
`penalty_dist` / `offdiag_cv` power diagnostics.

## Cost estimate

Per array task: 1 baseline + 12 regularized configs × 10 runs. Dominated
by amazon-ratings (2500 epochs × 130 runs ≈ heavy but each epoch is fast)
and roman-empire/GAT. Everything fits comfortably in the 48 h limit per
task; the whole experiment is one submission on 8 GPUs.
