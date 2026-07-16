#!/bin/bash -l
#SBATCH --job-name=gt_reg
#SBATCH --output=gt_reg_%A_%a.log
#SBATCH --error=gt_reg_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-51
#SBATCH --requeue

# GT regularization sweep + GT placebo (EXPERIMENT_PLAN E2.2 + E4.1-GT) on the
# 13 published-config combos. Baselines are NOT re-run: they come from
# submit_gt_tuned.sbatch, which writes paired-seeded baseline rows into
# results_tuned/ — the same result_dir used here, so run k pairs exactly.
# SUBMIT THIS WITH A DEPENDENCY ON THE gt_tuned JOB (see submit_all.sh).
#
# Each combo is split into four condition groups (5 configs each) to keep
# the heaviest polynormer tasks (2500+ global epochs) inside the 48h limit:
#   group 0: mlp/none      group 1: mlp/shuffle
#   group 2: mlp/homophily group 3: oracle
# task_id = combo_idx * 4 + group
#
# Must be submitted from GTs_baselines/.

set -e

DEVICE=0
RUNS=10
EXTRA_BASE="--result_dir results_tuned"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"
nvidia-smi

COMBOS=(
  "polynormer roman-empire"
  "polynormer amazon-ratings"
  "polynormer minesweeper"
  "polynormer questions"
  "polynormer amazon-computer"
  "polynormer amazon-photo"
  "polynormer coauthor-cs"
  "polynormer coauthor-physics"
  "polynormer wikics"
  "sgformer cora"
  "sgformer citeseer"
  "sgformer pubmed"
  "sgformer chameleon"
)

COMBO_IDX=$((SLURM_ARRAY_TASK_ID / 4))
GROUP=$((SLURM_ARRAY_TASK_ID % 4))
read -r MODEL DATASET <<< "${COMBOS[$COMBO_IDX]}"
echo "Reg sweep: $MODEL / $DATASET | group $GROUP"

# isolate checkpoints: group 0 and 1 for the same combo run concurrently and
# polynormer's phase switch reloads from disk
EXTRA="$EXTRA_BASE --model_dir models/gt_reg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

CMD=$(grep -m1 -- "--dataset $DATASET " ${MODEL}_tuned.sh || true)
if [ -z "$CMD" ]; then
    echo "ERROR: no command for $MODEL/$DATASET in ${MODEL}_tuned.sh"; exit 1
fi
echo "$CMD"

case $GROUP in
  0) ../sweep_conditions.sh "$(eval echo "$CMD")" results_tuned "none"      0 0 ;;
  1) ../sweep_conditions.sh "$(eval echo "$CMD")" results_tuned "shuffle"   0 0 ;;
  2) ../sweep_conditions.sh "$(eval echo "$CMD")" results_tuned "homophily" 0 0 ;;
  3) ../sweep_conditions.sh "$(eval echo "$CMD")" results_tuned ""          1 0 ;;
esac

rm -rf "models/gt_reg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Task $SLURM_ARRAY_TASK_ID done."
