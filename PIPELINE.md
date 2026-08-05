# Unified Experiment Pipeline

One submission (`./submit_all.sh` from the repo root on Wulver) runs every
experiment needed to redo the core results under the fixed protocol. SLURM
dependencies enforce the order; the placebo runs first.

```
                      ┌─────────────────────────────────────────────┐
                      │ 1. PLACEBO (submit_placebo.sbatch, 48 tasks│
                      │ 8 headline GNN pairs × {baseline, mlp×3     │
                      │ transforms×5λ, oracle×5λ, degraded-MLP      │
                      │ quality curve ep∈{10,50}×5λ} × 10 runs      │
                      └──────┬──────────────┬──────────────┬────────┘
                    afterany │              │              │
        ┌────────────────────▼───┐  ┌───────▼─────────┐  ┌─▼──────────────────┐
        │ 2. GNN UNIFIED SWEEP   │  │ 3. GT TUNED     │  │ 4. GT CONFIG SEARCH│
        │ (84 tasks, 68 active)  │  │ BASELINES (13)  │  │ (15 gap combos)    │
        │ remaining run_gnn.sh   │  │ published       │  │ bounded grids,     │
        │ pairs × {baseline,     │  │ configs, 10     │  │ 3 runs/config      │
        │ mlp none×5λ, oracle×5λ}│  │ paired runs     │  │                    │
        └────────────────────────┘  └───────┬─────────┘  └─┬──────────────────┘
                                   afterany │     afterany │
                                    ┌───────▼─────────┐  ┌─▼──────────────────┐
                                    │ 5. GT REG SWEEP │  │ 6. GT WINNER       │
                                    │ (52 tasks)      │  │ CONFIRM (15 tasks) │
                                    │ mlp×3 transforms│  │ 10-run baselines   │
                                    │ ×5λ + oracle×5λ │  │ at searched configs│
                                    │ on tuned GTs    │  └─┬──────────────────┘
                                    └─────────────────┘    │ afterany
                                                         ┌─▼──────────────────┐
                                                         │ 7. GT REG SWEEP ON │
                                                         │ SEARCH WINNERS (60)│
                                                         │ completes E2.2 for │
                                                         │ all 28 GT combos   │
                                                         └────────────────────┘
```

## What each stage answers

| Stage | Plan item | Question |
|---|---|---|
| 1 Placebo | E4.1 | Is the GNN gain co-occurrence *information* or generic regularization? (shuffle + homophily controls, power diagnostics) |
| 2 GNN unified | E2.1 + E2.3 | Honest mlp_gnn table under validation selection + paired stats; oracle ceiling per pair. Uses `run_gnn.sh` lines verbatim (fixes roman-empire/GCN config and keeps rocauc metrics). |
| 3 GT tuned | E1.1 | Do the published GT configs reproduce published numbers here? **Acceptance gate for stages 5's interpretation.** |
| 4 GT search | E1.1 | Baseline configs for the 15 (model, dataset) pairs with no published config. |
| 5 GT reg sweep | E2.2 + E4.1-GT | Does co-occurrence regularization still beat *properly tuned* GTs? Includes the GT placebo and oracle. **The decisive experiment.** |
| 6 GT confirm | E1.1 | 10-run validation of each search winner, written into `results_tuned/`. |
| 7 GT reg (searched) | E2.2 | Reg sweep + placebo + oracle on the 15 search-winner combos — E2.2 then covers all 28 (GT, dataset) pairs. |

## Protocol invariants (all stages)

- λ grid **{0.01, 0.05, 0.1, 0.2, 0.4}** (0.01 restored — several original
  wins peaked there).
- `--paired_seeds` everywhere: run *k* identical across conditions.
- λ/config selection on **validation only** (`analyze_placebo.py`,
  `analyze_gt_search.py`).
- Per-run CSVs (`runs_*.csv`) are the analysis substrate; aggregate CSVs are
  legacy.
- Checkpoint dirs isolated per SLURM task (`--model_dir`), cleaned on exit.
- Heavy (dataset, model) pairs are split into per-condition-group tasks
  (max ~150k training epochs per task) so no task can exceed the 48h limit;
  cross-group pairing holds because `--paired_seeds` makes run *k*
  deterministic from the seed alone.
- `--requeue`: sweeps are **resumable** — each completed condition leaves a
  stamp under `<result_dir>/.done/`, so a preempted task skips finished
  conditions on restart (delete `.done/` to force a re-run). Analyzers
  additionally dedupe by (condition, seed, run), keeping the last row.

## After completion

```bash
cd medium_graph
python analyze_placebo.py --result_dir results/placebo    # stage 1 verdict
python analyze_placebo.py --result_dir results/placebo results/unified  # full E2.1 table (merged)
cd GTs_baselines
python ../analyze_placebo.py --result_dir results_tuned   # stage 3+5 tables
python analyze_gt_search.py                                # stage 4 winners
```

Gates before believing anything downstream (automated):
```bash
cd medium_graph
python check_baselines.py --result_dir results/placebo results/unified
python check_baselines.py --result_dir GTs_baselines/results_tuned
```
`check_baselines.py` FAILs any GT baseline >1.5 pts below its published
target and flags any baseline (GNN or GT) sitting far below the best model
on the same dataset (the roman-empire/GCN failure mode). Additionally:
`penalty_dist` must be ≫ 0 wherever a shuffle/homophily verdict is claimed —
the homophily transform is mathematically vacuous on binary datasets (its
sweep groups are skipped there).

## Follow-up stage now available: low-label experiment (E3.1)

`medium_graph/submit_lowlabel.sbatch` + `submit_lowlabel_mv.sbatch` —
independent of this pipeline (uses its tuned configs and its gt_search
results); see `medium_graph/LOWLABEL_README.md`.

## Not in this submission (follow-ups)

- SBM recovery (E3.2), gradient alignment (E3.3) — separate scripts,
  independent of this pipeline.
- Scale-normalization ablation (E4.2) — needs a fixed-λ mode in main.py and
  its own λ grid.
- gnn_gnn (dynamic penalty) re-run — deprioritized in the plan; FINDINGS §5
  remains on old-protocol data until then.
- Extending the E3.4 cooc-quality diagnostic beyond 4 GCN datasets.
- ogbn-arxiv (large_graph/) under the corrected protocol (E4.4).
- NodeFormer (blocked on cluster nvcc).

Note on chameleon/squirrel: the loaders already read the **filtered**
Platonov versions (`*_filtered.npz`) — P0.4 is satisfied; the node counts in
FINDINGS.md Table 1 (2,277/5,201) describe the old versions and are stale.
