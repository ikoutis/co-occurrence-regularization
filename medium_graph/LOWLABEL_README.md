# Low-Label Experiment (E3.1): Does the Co-occurrence Prior Substitute for Labels?

The full-supervision study (FINDINGS_V2.md) established: the prior is worth
~zero when labels are plentiful, its only nonzero oracle ceilings sit on the
20-labels/class datasets (citeseer +2.5, cora +1.1), and gains track how
well the co-occurrence matrix can be estimated. This experiment tests the
resulting hypothesis directly:

> **As the label budget shrinks, the value of the co-occurrence prior
> grows — and so does the difficulty of estimating it.**

## Design

**Grid:** 8 datasets (cora, citeseer, pubmed, amazon-computer, coauthor-cs,
wikics, amazon-ratings, roman-empire) × {GCN, Polynormer} × budgets
{2, 5, 10, 20, 50} labels/class × 10 paired runs.

**Hyperparameters are the full-supervision tuned configs, frozen** (run_gnn.sh
for GCN; polynormer_tuned.sh published configs / gt_search winners for
Polynormer). The paired Δ is config-fair — baseline and every condition share
the config — and freezing removes the asymmetric-tuning bias. The 20/50-budget
cells connect to the full-supervision anchor points.

**Budget mechanics:**
- cora/citeseer/pubmed (`rand_split_class` protocol): `--label_num_per_class N`
  with `--resample_split_per_run` — each run draws a fresh label sample,
  seeded by `seed + run`, so every condition sees the identical sample
  (split-sampling variance is inside the paired statistics, not ignored).
- all others (fixed splits): `--train_per_class N` — the fixed train split is
  subsampled per run (seeded); val/test untouched, so curves join the anchor.

**Three penalty estimators + controls, λ ∈ {0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6}**
(extended upward: with a weak CE term the optimal prior weight can exceed the
full-supervision grid):

| Condition | Source | Leaks? |
|---|---|---|
| `count` | class-pair counts on **train-train edges** (+Laplace smoothing) | no |
| `mlp` | features-only MLP trained on the budget's labels | no |
| `oracle` | all labels incl. test | **yes — diagnostic upper bound only** |
| `shuffle` | permuted MLP penalty (budgets 2, 5) | placebo |

Per-run covariates logged: `cooc_oracle_dist` (estimate's distance to the
all-label matrix — figure 2's y-axis), `n_prior_edges` (the count
estimator's honest sample size), `mlp_acc`, `budget`.

**Leakage guards:** the MLP trains on `train_idx` *after* budget subsampling
(the subsample happens at the top of the run loop, before any penalty
construction); the count estimator sees only labels of train nodes on
train-train edges. The all-label oracle is intentionally leaky and labeled
as such everywhere.

**Matched-budget validation check** (`submit_lowlabel_mv.sbatch`): at budgets
{2, 5} the standard 500-node validation set carries more supervision than the
train set — the classic low-label evaluation pitfall. A parallel run with
`--valid_per_class` = train budget re-does baseline+mlp; if λ* and Δ agree
with the main run, the curves stand on either protocol.

## Predictions (falsifiable)

1. Baseline-vs-regularized curves converge at budget 50 and separate as the
   budget shrinks (a crossing point per dataset).
2. The oracle gap grows monotonically as budget shrinks.
3. `count` beats `mlp` where train-train edges are dense (high-budget
   heterophilic cells) and collapses to uniform (dist→1, n_edges→0) at tiny
   budgets, where `mlp` — if anything — must carry the signal.
4. Shuffle stays at zero everywhere (if not, small-budget gains are generic
   regularization, and the story dies honestly).

## Running

```bash
cd medium_graph
sbatch submit_lowlabel.sbatch        # main grid: array 0-399 (48 no-op tasks)
sbatch submit_lowlabel_mv.sbatch     # matched-val check: array 0-63
# afterwards:
python analyze_lowlabel.py                                        # curves
python analyze_lowlabel.py --result_dir results/lowlabel_mv       # sensitivity
```

Both arrays are requeue-resumable (condition stamps under
`<result_dir>/.done/`).

## Cost

Heaviest tasks (amazon-ratings/roman-empire × Polynormer, 2500+ epochs):
≤7 conditions × 10 runs × ~2700 epochs ≈ 189k training epochs per task —
inside 48 h with margin, and requeue-resume covers any residual overrun.
Most tasks are far smaller (cora GCN cells finish in minutes).
400 + 64 tasks total, one submission each.
