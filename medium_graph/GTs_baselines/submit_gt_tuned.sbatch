#!/bin/bash -l
#SBATCH --job-name=gt_tuned
#SBATCH --output=gt_tuned_%A_%a.log
#SBATCH --error=gt_tuned_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-12
#SBATCH --requeue

# Phase 1 baseline verification (EXPERIMENT_PLAN.md E1.1): run the PUBLISHED
# GT configs and check they reproduce the published numbers before any
# regularization experiment uses them. Acceptance gate: within ~1 point of
# the targets listed in polynormer_tuned.sh / sgformer_tuned.sh.
#
# Each array task runs one (model, dataset): the command line is extracted
# from the corresponding *_tuned.sh so there is a single source of truth.
#
# Must be submitted from GTs_baselines/.

set -e

DEVICE=0
RUNS=10
EXTRA="--result_dir results_tuned --paired_seeds"

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

read -r MODEL DATASET <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
echo "Verifying published config: $MODEL / $DATASET"

CMD=$(grep -m1 -- "--dataset $DATASET " ${MODEL}_tuned.sh)
if [ -z "$CMD" ]; then
    echo "ERROR: no command for $MODEL/$DATASET in ${MODEL}_tuned.sh"; exit 1
fi
echo "$CMD"
eval "$CMD"

echo "Done: $MODEL / $DATASET"
