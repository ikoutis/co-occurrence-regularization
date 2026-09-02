# Transfer Experiment (E4): A Legitimate Source for the Co-occurrence Matrix

The low-label study (LOWLABEL_README.md, results/lowlabel) showed that the
*oracle* K×K matrix is worth 1.4–4× the label budget at 10–20 labels/class,
but that estimating it from the budget itself (`count`, `mlp`) recovers only a
small part of that. The natural objection: *the oracle needs labels you do not
have*. This experiment answers it with a source the practitioner **does**
have — the labeled past of a temporal graph.

## Design

**Dataset:** ogbn-arxiv (40 classes, per-node year). Temporal split at
`YEAR = 2018`:

- **source pool** = papers with year < 2018 (~91k, fully labeled — it is the past)
- **target pool** = papers with year ≥ 2018 (~78k); *all* train/valid/test
  nodes are drawn from here, per class, with a fresh sample every run
  (`--rand_split_class --label_num_per_class N --resample_split_per_run`,
  valid 2000 / test 20000).

**The prior** (`make_cooc_prior.py`) is the class-pair count matrix on
source–source edges (Laplace-smoothed, row-normalized) — the `count`
estimator applied to the past. It is saved once and loaded with
`--cooc_file` (`reg_type = transfer` in the runs CSV). It never sees a target
node. Two scarce-past variants (`--source_per_class 100 / 20`) ask how much
past is needed.

**Three source modes** (`--source_mode`):

| mode | graph | training labels | what it tests |
|---|---|---|---|
| `keep` | whole graph; source nodes present but unlabeled | N/class from target | the realistic transductive setting: history is in the graph, only its K×K summary reaches the loss |
| `drop` | induced subgraph on the target pool | N/class from target | the prior is the *only* thing that crosses the split (no source features, no source edges) |
| `labels` | whole graph | N/class from target **+ all source labels** | the "just use the labels" reference — what someone who has the past's labels, not only their summary, would do |

**Conditions per cell** (λ ∈ {0.01 … 1.6}, 10 paired runs, validation-selected λ,
Wilcoxon on paired deltas — the same protocol as everything else):

- `keep`, `drop`: baseline · count · mlp · oracle† · **transfer** · transfer/shuffle (placebo)
- `keep` also: transfer from 100/class and 20/class of the past
- `labels`: baseline · transfer (does the prior add anything on top of the labels?)

† oracle = all labels incl. test (and, in keep/labels, the past) — leaky diagnostic.

**Budgets:** 5, 20, 50 labels/class (200 / 800 / 2000 training labels).

**Model:** GCN with the tunedGNN ogbn-arxiv config, frozen (hidden 512,
5 layers, bn, res, lr 5e-4, 2000 epochs). Polynormer is left out: full-batch
on 169k nodes is a memory/time risk on the 20 GB cards and adds nothing to the
question this experiment asks.

## What the outcomes would mean

- transfer ≫ shuffle ≈ 0, and transfer ≈ oracle: the prior *is* learnable
  from history and survives temporal drift → the paper's positive claim.
- transfer > count/mlp: the past is a better estimator than the budget —
  the "hard to estimate from scarce labels" story, with a remedy.
- `labels` baseline ≫ `keep`+transfer: the K×K summary is worth much less
  than the labels themselves — expected; the honest framing is then *when
  only the summary is available* (privacy, a related but disjoint graph,
  a different feature space — `drop` mode is the proxy).
- transfer on top of `labels` ≈ 0: the prior carries no information beyond
  the labels it was counted from — also expected, and fine.
- transfer ≈ 0 while oracle > 0: the matrix drifts too much across years
  (check `prior_dist` — the logged distance of the prior to the target-only
  oracle matrix) → the claim must be qualified to stationary graphs.

## Running

```bash
cd medium_graph
# 1. build the priors once (login node; downloads ogbn-arxiv if not found —
#    pass --data_dir ../large_graph/data/ if the large_graph copy exists)
python make_cooc_prior.py --year_split 2018 --out priors/arxiv_lt2018_all.pt
python make_cooc_prior.py --year_split 2018 --source_per_class 100 --out priors/arxiv_lt2018_m100.pt
python make_cooc_prior.py --year_split 2018 --source_per_class 20  --out priors/arxiv_lt2018_m20.pt
# 2. the grid (48 tasks, requeue-resumable)
sbatch submit_transfer.sbatch
# 3. afterwards
python analyze_transfer.py
```

## Cost

One task = ≤7 configs × 10 runs × 2000 full-batch epochs on 169k nodes
(78k in `drop` mode) ≈ 6–8 h on an A100-20G; 48 tasks. The `labels` cells
train on ~93k labels and converge much earlier, but keep the frozen
schedule for comparability.
