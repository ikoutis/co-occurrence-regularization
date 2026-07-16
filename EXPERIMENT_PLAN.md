# Experiment Plan: From Exploratory Results to Publishable Evidence

This plan addresses the methodological issues in the current results
(FINDINGS.md) and lays out the experiments needed to either confirm or
kill each claim. Phases are ordered by dependency: Phase 0–1 are
prerequisites — re-running sweeps before fixing the protocol would waste
compute.

**The two problems this plan is built around:**

1. **λ-selection noise.** Current Δ values are max-over-a-4-point-λ-grid
   selected on *test* accuracy, with effect sizes (+0.3–1.0) comparable to
   run-to-run std (0.5–0.9). The tell: mlp_gnn "beats" the oracle on several
   rows (squirrel/GAT, cora/GAT, citeseer/GAT), which is impossible except
   as selection noise.
2. **Undertuned baselines.** The GT full sweep
   (`submit_sweep_mlp_gt_full.sbatch`) used one generic config
   (hidden 128, 2 local + 2 global layers, 4 heads) for all 14 datasets.
   Result: Polynormer baseline 70.16 on amazon-computer vs ~93.7 published.
   The headline "+3.94, 39× the GNN gain" is a gain on a broken baseline.
   Same class of problem on the GNN side: roman-empire/GCN at 28.54 vs
   ~88–91 in the tunedGNN paper this repo forks.

---

## Phase 0 — Protocol and code fixes (no GPU time, do first)

### P0.1 Validation-based λ selection + paired statistics

**Change:** For each (dataset, model), select λ* by **mean validation
accuracy** across runs; report the test accuracy at that λ*. Never select
on test.

**Change:** Make baseline and regularized runs **paired**: same seed →
same split, same model init, same data order. Then report:
- mean ± std of test accuracy at validation-selected λ*
- paired difference Δ with a Wilcoxon signed-rank test (or paired t-test)
  across runs
- effect flagged as real only if p < 0.05 *and* |Δ| > pooled std/√runs

**Code:**
- `main.py`: add `--seed_offset` so run *k* uses seed `seed + k` for both
  model init and split selection (verify `split_idx_lst[run]` and
  `reset_parameters()` are seed-deterministic).
- New `analyze_results.py`: reads `results/`, does validation-based λ
  selection, paired stats, emits the FINDINGS tables automatically.
  No more hand-assembled tables — this also removes transcription errors.

### P0.2 Runs and grid

- 10 runs (up from 5) for every headline configuration. Keep 5 for
  exploratory sweeps.
- λ grid: {0.01, 0.05, 0.1, 0.2, 0.4}. The extra midpoint costs 25% more
  but removes the "coarse grid" excuse used to explain oracle anomalies.

### P0.3 Metrics

- `questions`, `minesweeper`, (and `tolokers` if added): report **ROC-AUC**
  everywhere, matching Platonov et al. and tunedGNN. The current 97%
  "accuracy" on questions is class-imbalance saturation and hides any
  effect. The GT scripts already pass `--metric rocauc`; the GNN sweep
  does not — unify.

### P0.4 Datasets

- ~~Replace chameleon/squirrel with the filtered versions~~ **Already
  satisfied**: the loaders read `*_filtered.npz` (Platonov et al. filtered
  versions). The node counts in FINDINGS.md Table 1 (2,277/5,201) describe
  the old versions and are stale documentation, not what was run.

### P0.5 Fix degenerate baselines before interpreting them

- **roman-empire/GCN (28.54):** broken config, not a hard dataset —
  tunedGNN reports ~88–91. Diagnose (likely missing `--res`/`--ln` or
  wrong depth), fix, re-run. Until fixed, delete this row from all
  conclusions.
- **questions/GCN:** currently predicts majority class. Same treatment.

### P0.6 Housekeeping

- FINDINGS.md: section 8 is referenced but missing (numbering skips 7→9).
- Log, for every run: MLP accuracy, and distance of the penalty-source
  co-occurrence matrix to the oracle matrix (Frobenius + KL). These are
  the covariates for the "benefit scales with signal quality" claim —
  currently asserted from 3 data points.

---

## Phase 1 — Re-establish baselines (gate for everything downstream)

### E1.1 GT baselines at published hyperparameters

**Status: IMPLEMENTED** — see `GTs_baselines/GT_BASELINES_README.md`.

Investigation revealed the baseline gap was not only hyperparameters:
(a) Polynormer's two-phase schedule was missing from the adapted `main.py`,
so `model._global` stayed False and **the global attention module was never
trained** in any prior Polynormer run; (b) SGFormer's transformer depth was
tied to its GCN backbone depth (official configs use 1 transformer layer);
(c) amazon/coauthor splits differed between the GT and GNN sweeps. All
three fixed. Published per-dataset configs pulled from the official repos
(`polynormer_tuned.sh`, `sgformer_tuned.sh`); the (model, dataset) pairs
with no published config get a bounded validation-selected search
(`submit_gt_search.sbatch` + `analyze_gt_search.py`).

**Acceptance gate:** 10-run mean within ~1 point of published numbers
(Polynormer amazon-computer 94.07, roman-empire 92.48, …; SGFormer cora
84.5, citeseer 72.6, pubmed 80.3). If a dataset can't be matched, find out
why before running any regularization on it.
Run: `sbatch submit_gt_tuned.sbatch` + `sbatch submit_gt_search.sbatch`.

### E1.2 GNN baselines cross-checked against tunedGNN

Compare every baseline row of Table 2 against the tunedGNN paper. Flag
and fix any gap > 1.5 points (roman-empire/GCN is the known case).

**Cost:** ~30 GPU-days equivalent on the a100_20g queue, mostly Phase 1.1
(2500-epoch Polynormer runs). Parallelizes fully as an sbatch array.

---

## Phase 2 — Main sweeps under the fixed protocol

### E2.1 mlp_gnn re-run

14 datasets × {GCN, GAT, SAGE} × λ grid × 10 paired runs, analyzed by
`analyze_results.py`. Expectation: the "small but consistent" wins
(+0.3–0.5) mostly collapse; the larger ones (cora, citeseer, squirrel,
amazon-ratings) may survive. Either outcome is fine — we need the honest
table.

### E2.2 mlp_gt re-run on tuned baselines — **the decisive experiment**

14 datasets × {Polynormer, SGFormer} × λ grid × 10 paired runs, starting
from E1.1 configs. NodeFormer if the nvcc/torch_sparse issue is resolved
(try a pre-built wheel matching the cluster's CUDA runtime, or CPU-compile
once on the login node); otherwise drop it and say so.

**This is the experiment the paper lives or dies on.** Two outcomes:
- Gains survive on tuned GTs (even at +0.5–1.5 instead of +3.9): the
  headline claim stands, now defensible.
- Gains vanish: the "GT gains" were compensation for undertraining.
  The paper pivots to the mechanism + label-efficiency story (Phase 3),
  which is publishable on its own.

### E2.3 Oracle under the same protocol

Same grid, oracle penalty, 10 paired runs, on the ~8 datasets where E2.1/2.2
show any effect. Report "fraction of oracle gap recovered" with error bars.
Under validation-based selection, mlp should no longer beat oracle; if it
still does, something is wrong — stop and investigate.

---

## Phase 3 — The science (mechanism + regime where the method matters)

These are the experiments that make it a paper rather than a benchmark note.
They can start as soon as Phase 0 is done, in parallel with Phase 2.

### E3.1 Label efficiency (OPEN_QUESTIONS Q4) — highest value

- Datasets: cora, citeseer, pubmed, amazon-computer, coauthor-cs, wikics
  (homophilic, where full-supervision gains are ~0).
- Vary labels/class ∈ {2, 5, 10, 20, 50}; fixed val/test.
- Models: GCN and Polynormer, each ×{baseline, mlp_reg, oracle_reg},
  validation-selected λ, 10 paired runs per point.
- **Important control:** the penalty-source MLP must be trained on the
  *same reduced* label set — otherwise the regularizer smuggles in labels
  the baseline doesn't have.
- Deliverable: accuracy-vs-labels curves; the predicted crossing point
  (label count below which the regularizer helps even GNNs) is the
  paper's strongest practical figure.
- Cost: 6 datasets × 5 label counts × 2 models × 3 conditions ×
  (1 + λ grid) — big, but each run is small; ~1 week of array jobs.

### E3.2 SBM co-occurrence recovery (Q2) — the mechanism figure

`experiments/cooc_recovery/cooc_recovery.py` exists; extend to a grid:
- SBM: n=2000–5000, k ∈ {4, 8} blocks, homophilic and heterophilic B,
  3–4 SNR levels (vary p_in/p_out toward the detectability threshold).
- Models: GCN, pure attention GT (no positional encoding, no edge_index),
  MLP; co-occurrence loss only, no CE.
- Metrics: ‖C_pred − C*‖_F and Hungarian-matched clustering accuracy.
- The hypothesized signature — GT reaches low Frobenius error with
  chance-level clustering accuracy (degenerate solution) while GCN gets
  both — is the cleanest possible demonstration of *why* topology-free
  attention needs the explicit penalty. Cheap: CPU-scale, days not weeks.

### E3.3 Gradient alignment

During training on 4 representative datasets (cora, amazon-computer,
amazon-ratings, roman-empire) × {GCN, Polynormer}: log
cos(∇L_CE, ∇L_reg) per epoch (both gradients w.r.t. shared parameters).
Prediction from the redundancy story: alignment high/positive for GCN
(penalty redundant), near zero or negative early for Polynormer (penalty
informative). One extra backward pass per logged epoch; negligible cost.

### E3.4 Co-occurrence quality diagnostic (Q1) — extend

`measure_cooc_quality.py` currently covers 3 datasets, GCN only. Extend to
8 datasets × {GCN, Polynormer}. The cora result (Frobenius error to oracle
C falls monotonically with λ while accuracy rises) is the direct
validation of the mechanism — one dataset isn't enough to print it.

### E3.5 Two-phase training (Q3) — run only if E2.2 survives

Warm-start variant only (phase 1: edge_loss; phase 2: CE + λ·edge_loss).
2 datasets × Polynormer as a pilot before any wider sweep.

---

## Phase 4 — Ablations reviewers will demand

### E4.1 Placebo control (cheap, high evidential value — run early)

**Status: IMPLEMENTED** — see `medium_graph/PLACEBO_README.md`,
`submit_placebo.sbatch` (single Wulver submission, array 0–7).

Design refined from the original sketch after noting that a permuted
penalty is not a clean placebo when co-occurrence statistics are similar
across class pairs (and that class-relabeling permutations preserve
homophily and test nothing). Final design: `--penalty_transform
{shuffle, homophily}` — a full entry shuffle (destroys all semantics) and
a homophily-only control (keeps the diagonal, removes class-pair
structure) — plus a per-run distinguishability diagnostic
(‖P_shuffled − P_true‖_F / ‖P_true‖_F) so rows where the transform barely
changed the penalty are flagged as having no statistical power instead of
being misread. Paired runs via `--paired_seeds`; analysis with
validation-based λ selection and Wilcoxon tests in
`medium_graph/analyze_placebo.py`.

### E4.2 Scale normalization ablation

The per-step `scale = task/reg` normalization makes the effective weight
non-stationary and reviewers will ask. Compare on 4 datasets:
(a) current adaptive scale, (b) fixed λ tuned on validation,
(c) scale frozen after epoch 50. If (b) matches (a), drop the
normalization — simpler method, simpler paper.

### E4.3 Penalty-source quality curve

Vary MLP quality directly (mlp_epochs ∈ {10, 50, 500}; or train the MLP on
{25%, 50%, 100%} of train labels) on 4 datasets. Plot Δ vs penalty-matrix
distance-to-oracle. Turns claim #5 ("benefit scales with signal quality")
from anecdote into a fitted curve, using the covariates logged in P0.6.

### E4.4 Large-graph check

ogbn-arxiv runs exist (`large_graph/results/arxiv_*`); bring them under the
same protocol (validation selection, paired runs). One large-graph point
with the corrected protocol is enough for the paper; products/proteins
optional.

---

## Sequencing and decision points

```
Phase 0 (protocol/code)          ~1 week, no GPU
   │
Phase 1 (baselines)              gate: match published numbers
   │
   ├─ Phase 2 (main sweeps)      DECISION at E2.2:
   │     GT gains survive → headline = GT claim
   │     GT gains vanish  → headline = mechanism + label efficiency
   │
   ├─ Phase 3 (E3.1 label efficiency, E3.2 SBM, E3.3 gradients)
   │     — start in parallel with Phase 2, independent of its outcome
   │
   └─ Phase 4 (E4.1 placebo EARLY; rest after Phase 2)
```

Priority order if compute is tight:
1. P0.* + E1.1 (nothing else is interpretable without them)
2. E2.2 (decisive) + E4.1 (placebo — cheap insurance)
3. E3.1 (label efficiency) and E3.2 (SBM) — the paper's figures
4. E2.1/E2.3 refresh, E3.3, E3.4, E4.2, E4.3
5. E4.4, E3.5

## What gets claimed, and what kills each claim

| Claim | Supported if | Killed if |
|---|---|---|
| Co-occ reg helps GTs beyond GNN ceiling | E2.2 gains > 0 (paired, p<.05) on tuned baselines | gains vanish on tuned baselines |
| Mechanism is co-occurrence info, not generic smoothing | E4.1 permuted penalty ≈ baseline | permuted penalty also helps |
| GTs lack topology bias that GNNs have | E3.2 GT: low Fro / chance clustering; GCN: both good | GT also recovers clustering |
| Method matters when labels are scarce | E3.1 crossing point exists | curves parallel at all label counts |
| Benefit tracks penalty quality | E4.3 monotone Δ vs matrix quality | flat relationship |
