#!/bin/bash

# Master submission for the full experiment pipeline (see PIPELINE.md).
# One command; SLURM dependencies enforce the ordering:
#
#   placebo (48 tasks) ─┬─> unified GNN sweep   (84 tasks, 68 active)
#                       ├─> GT tuned baselines  (13 tasks) ──> GT reg sweep (52 tasks)
#                       └─> GT config search    (15 tasks) ──> GT winner confirm (15)
#                                                                    └─> GT reg sweep on
#                                                                        search winners (60)
#
# 'afterany' is used (not 'afterok') so one crashed task cannot leave the
# rest of the pipeline pending forever; failed tasks show up in the logs and
# in missing CSV rows.
#
# Usage: ./submit_all.sh          (from the repo root)

set -e
cd "$(dirname "$0")"

command -v sbatch >/dev/null || { echo "sbatch not found — run this on the cluster login node."; exit 1; }

# preflight: warn about dataset dirs the pipeline expects
for d in medium_graph/data/geom-gcn medium_graph/data; do
    [ -e "$d" ] || echo "WARNING: $d not found — run the dataset download steps first (see medium_graph/download_geom_gcn.sh and dataset.py auto-downloads)."
done

echo "Submitting pipeline..."

cd medium_graph
J_PLACEBO=$(sbatch --parsable submit_placebo.sbatch)
echo "  placebo            : $J_PLACEBO  (48 tasks: 8 headline GNN pairs x 6 condition groups)"

J_GNN=$(sbatch --parsable --dependency=afterany:$J_PLACEBO submit_unified_gnn.sbatch)
echo "  gnn unified sweep  : $J_GNN  (after $J_PLACEBO; remaining 34 pairs x 2 groups, mlp+oracle)"

cd GTs_baselines
J_GT_TUNED=$(sbatch --parsable --dependency=afterany:$J_PLACEBO submit_gt_tuned.sbatch)
echo "  gt tuned baselines : $J_GT_TUNED  (after $J_PLACEBO; 13 published configs, 10 runs)"

J_GT_SEARCH=$(sbatch --parsable --dependency=afterany:$J_PLACEBO submit_gt_search.sbatch)
echo "  gt config search   : $J_GT_SEARCH  (after $J_PLACEBO; 15 gap combos)"

J_GT_REG=$(sbatch --parsable --dependency=afterany:$J_GT_TUNED submit_gt_reg_sweep.sbatch)
echo "  gt reg sweep       : $J_GT_REG  (after $J_GT_TUNED; reg+placebo+oracle on tuned GTs)"

J_GT_CONFIRM=$(sbatch --parsable --dependency=afterany:$J_GT_SEARCH submit_gt_confirm.sbatch)
echo "  gt winner confirm  : $J_GT_CONFIRM  (after $J_GT_SEARCH; 10-run confirmation)"

J_GT_REG_S=$(sbatch --parsable --dependency=afterany:$J_GT_CONFIRM submit_gt_reg_search.sbatch)
echo "  gt reg (searched)  : $J_GT_REG_S  (after $J_GT_CONFIRM; reg+placebo+oracle on search winners)"

cd ../..

cat <<EOF

All submitted. Watch with:  squeue -u \$USER
When everything finishes, analyze:
  cd medium_graph
  python analyze_placebo.py --result_dir results/placebo      # placebo verdict
  python analyze_placebo.py --result_dir results/unified      # E2.1/E2.3 GNN tables
  cd GTs_baselines
  python ../analyze_placebo.py --result_dir results_tuned     # E2.2 GT tables + GT placebo
  python analyze_gt_search.py                                 # search winners

Acceptance gates BEFORE trusting downstream numbers:
  - GT tuned baselines within ~1 pt of published (GT_BASELINES_README.md)
  - roman-empire/GCN baseline should now be ~88-91, not 28.5
EOF
