#!/bin/bash -l
#SBATCH --job-name=gt_reg_s
#SBATCH --output=gt_reg_s_%A_%a.log
#SBATCH --error=gt_reg_s_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-59
#SBATCH --requeue

# GT regularization sweep on the 15 SEARCH-WINNER combos — together with
# submit_gt_reg_sweep.sbatch (the 13 published-config combos) this completes
# E2.2 over all 28 (GT, dataset) pairs. The winner's base command is
# reconstructed from the search CSVs by gt_confirm_cmd.py (validation-
# selected); baselines come from submit_gt_confirm.sbatch, which wrote
# paired-seeded baseline rows into results_tuned/ for the same configs.
# SUBMIT WITH A DEPENDENCY ON THE gt_confirm JOB (see submit_all.sh).
#
# task_id = combo_idx * 4 + group
#   group 0: mlp/none      group 1: mlp/shuffle
#   group 2: mlp/homophily group 3: oracle
#
# Must be submitted from GTs_baselines/.

set -e

DEVICE=0
RUNS=10

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"
nvidia-smi

COMBOS=(
  "polynormer cora"
  "polynormer citeseer"
  "polynormer pubmed"
  "polynormer chameleon"
  "polynormer squirrel"
  "sgformer amazon-computer"
  "sgformer amazon-photo"
  "sgformer coauthor-cs"
  "sgformer coauthor-physics"
  "sgformer wikics"
  "sgformer roman-empire"
  "sgformer amazon-ratings"
  "sgformer minesweeper"
  "sgformer questions"
  "sgformer squirrel"
)

COMBO_IDX=$((SLURM_ARRAY_TASK_ID / 4))
GROUP=$((SLURM_ARRAY_TASK_ID % 4))
read -r MODEL DATASET <<< "${COMBOS[$COMBO_IDX]}"
MODEL_DIR="models/gt_reg_s_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Reg sweep (search winner): $MODEL / $DATASET | group $GROUP"

# the homophily transform is mathematically the identity for 2-class
# datasets (one off-diagonal entry per row) — the condition is vacuous
if [ "$GROUP" = "2" ]; then
    case "$DATASET" in
        minesweeper|questions)
            echo "homophily transform is identity on binary datasets — skipping vacuous group."
            exit 0 ;;
    esac
fi

CMD=$(python gt_confirm_cmd.py --model $MODEL --dataset $DATASET \
      --runs $RUNS --device $DEVICE --model_dir $MODEL_DIR)
echo "$CMD"

case $GROUP in
  0) ../sweep_conditions.sh "$CMD" results_tuned "none"      0 0 ;;
  1) ../sweep_conditions.sh "$CMD" results_tuned "shuffle"   0 0 ;;
  2) ../sweep_conditions.sh "$CMD" results_tuned "homophily" 0 0 ;;
  3) ../sweep_conditions.sh "$CMD" results_tuned ""          1 0 ;;
esac

rm -rf "$MODEL_DIR"
echo "Task $SLURM_ARRAY_TASK_ID done."
