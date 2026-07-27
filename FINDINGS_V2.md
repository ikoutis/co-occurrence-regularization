# Findings V2: Full Re-run Under the Corrected Protocol

Results of the unified pipeline (PIPELINE.md): 10 paired runs, validation-only
λ selection, Wilcoxon signed-rank tests, λ ∈ {0.01, 0.05, 0.1, 0.2, 0.4},
placebo controls + oracle + penalty-quality curve, tuned GT baselines.
Raw per-run data: `results/placebo`, `results/unified`,
`GTs_baselines/results_tuned`, `GTs_baselines/gt_search`.

---

## 1. Baseline gates (check_baselines.py)

**GNN side: all 42 pairs pass.** The canonical `run_gnn.sh` configs fixed the
previously degenerate rows: roman-empire/GCN 28.54 → **90.89**, and
minesweeper/questions now report ROC-AUC (97.3 / 78.4). cora GCN reaches
84.38 — tunedGNN-level.

**GT side: Polynormer reproduces the paper; SGFormer is quarantined.**
All 9 published Polynormer configs land within 0.6 pts of the official
numbers (amazon-computer 93.68 vs 94.07, roman-empire 92.43 vs 92.48, …) —
the two-phase training fix worked. SGFormer FAILs its gates: citation
configs undershoot published by 1.7–3.9 pts, and the searched heterophilic
configs are clearly undertuned (roman-empire 74.7, minesweeper 80.9).
**SGFormer rows below are not interpretable as method evidence.**

---

## 2. The old headline claims, redone

| Claim (FINDINGS.md) | Old Δ | New Δ (paired, val-selected λ) | Verdict |
|---|---|---|---|
| squirrel/GAT | +2.23 | +0.69 (p=0.49) | gone (filtered data, huge variance) |
| citeseer/GAT | +1.26 | +0.00 (p=1.00) | gone |
| cora/GAT | +1.04 | +0.07 (p=0.72) | gone — and the **shuffled** penalty gives +0.71 (p=0.016) here, the generic-regularization signature |
| amazon-ratings/SAGE | +0.91 | **+0.44 (p=0.020)** | survives, halved |
| amazon-computer/Polynormer | +3.94 | −0.12 (p=0.10) | **dead** — was undertraining compensation |
| pubmed/Polynormer | +2.44 | −0.46 (p=0.33) | dead |
| coauthor-cs/Polynormer | +1.42 | +0.16 (p=0.055) | reduced to noise-level |
| coauthor-physics/SGFormer | +1.22 | −0.09 (p=0.014) | reversed (significantly negative) |

**The GT narrative ("GTs gain up to 40× more than GNNs") is dead.** On the
properly trained, properly tuned Polynormer, the regularization moves
nothing on the datasets that produced the old headline. The prior "GT
gains" were an artifact of (a) the never-enabled global attention phase,
(b) generic hyperparameters, and (c) test-set λ selection.

---

## 3. What survives: amazon-ratings, with a clean mechanistic signature

The single robust result, consistent across three architectures:

| Model | true Δ | homophily Δ | shuffle Δ | oracle Δ | shuffle power (pdist) |
|---|---|---|---|---|---|
| GCN | **+0.63** (p=.006) | +0.34 (ns) | 0.00 (ns) | +0.42 | 0.60 |
| SAGE | **+0.44** (p=.020) | +0.30 (ns) | +0.03 (ns) | +0.42 (p=.014) | 0.60 |
| Polynormer | **+0.61** (p=.014) | +0.63 (p=.010) | +0.19 (ns) | +0.67 (p=.004) | 0.59 |

- The shuffle placebo is powerful here (pdist ≈ 0.6) and lands at zero —
  the gain is **information**, not generic regularization.
- The penalty-quality curve (E4.3) is monotone on GCN:
  mlp_epochs 10 → 50 → 500 gives Δ = +0.28 → +0.49 → +0.63, tracking the
  penalty matrix's distance to the oracle matrix (0.46 → 0.51 → **0.12**).
- MLP recovers essentially the full oracle ceiling (+0.63 vs oracle +0.42–0.67).
- Nuance: on the GNNs, true > homophily (class-pair structure contributes);
  on Polynormer, homophily-only matches the full penalty — the GT gain is
  carried by the homophily level alone.

Smaller significant effects with the same direction: minesweeper GCN +0.22
(p=.027) / SAGE +0.51 (p=.049, oracle +0.41), pubmed/GCN +0.62 (p=.014, but
its oracle is ns — treat cautiously). questions shows tiny significant
**negative** effects (−0.08 to −0.12).

## 4. Where the placebo caught false positives

- **cora/GAT: shuffle +0.71 (p=.016) while true +0.07 (ns).** Any penalty
  of this form perturbs cora/GAT training enough to look like a gain at
  some λ. Effects of this size on cora are not method evidence.
- **citeseer/Polynormer: true +1.28 (p=.008) but shuffle +0.97 (p=.027)
  and homophily +1.06** — all three "help" a noisy, searched baseline
  (std 1.4). Generic regularization.
- **chameleon/Polynormer: shuffle −2.57 (p=.008)** — the penalty family
  can also actively hurt.

## 5. Oracle ceilings under the honest protocol

Largest oracle gains: citeseer (GCN +2.48, GAT +1.64, SAGE +1.04;
Polynormer cora +0.85). But the MLP-derived penalty captures **none** of
the citeseer ceiling (true Δ = 0.00) — and the covariates say why: the
citeseer MLP's co-occurrence matrix stays at relative distance ~0.80 from
the oracle matrix regardless of training (vs 0.12 on amazon-ratings).
**The method works exactly where a features-only model can estimate
co-occurrence well** — that is claim 5, now with a measured covariate
instead of an anecdote, and it doubles as the method's practical
limitation.

---

## 6. Implications for the paper

1. The standard-benchmark route is closed: surviving effects are +0.2–0.6
   on 2–3 datasets. mlp_gnn beating oracle never recurs under validation
   selection — the old anomalies were selection noise, as suspected.
2. The clean amazon-ratings triple (placebo-controlled, quality-monotone,
   oracle-matching) is a solid *mechanism* demonstration, not a headline.
3. The plan's pivot applies: the interesting open regime is
   **label efficiency (E3.1)** — where supervision is scarce and the
   co-occurrence prior has room to matter — plus the SBM recovery
   experiment (E3.2) for the mechanism figure.
4. There is also a publishable cautionary tale here: an undertrained
   baseline plus test-set λ selection manufactured a "+3.94, 40× GNN"
   result that a paired, validation-selected protocol on the tuned model
   reduces to −0.12.

## 7. Caveats / loose ends

- SGFormer: all rows quarantined (failed baseline gates). Its citation
  configs need diagnosis (likely remaining implementation deltas vs the
  official repo); its heterophilic search needs the tr_dropout /
  tr_weight_decay axes.
- coauthor-physics/SAGE: 1 of 84 unified tasks produced no CSV (task
  crashed or preempted past the window) — a near-zero-effect dataset;
  re-run its single array index if completeness matters.
- chameleon/squirrel: filtered datasets (890/2,223 nodes), std 2–4.5;
  nothing is significant there in either direction.
