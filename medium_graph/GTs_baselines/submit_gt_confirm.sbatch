#!/bin/bash -l
#SBATCH --job-name=gt_confirm
#SBATCH --output=gt_confirm_%A_%a.log
#SBATCH --error=gt_confirm_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-14
#SBATCH --requeue

# 10-run confirmation of each search winner (validation-selected by
# gt_confirm_cmd.py directly from the search CSVs — no shared state between
# array tasks). Writes paired-seeded baseline rows into results_tuned/, the
# same result_dir as the published-config baselines, so these combos become
# usable for a later reg sweep exactly like the published ones.
# SUBMIT WITH A DEPENDENCY ON THE gt_search JOB (see submit_all.sh).
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

read -r MODEL DATASET <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
MODEL_DIR="models/gt_confirm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Confirming search winner: $MODEL / $DATASET"

CMD=$(python gt_confirm_cmd.py --model $MODEL --dataset $DATASET \
      --runs $RUNS --device $DEVICE --model_dir $MODEL_DIR)
echo "$CMD"
eval "$CMD"

rm -rf "$MODEL_DIR"
echo "Confirmed: $MODEL / $DATASET"
