#!/bin/bash -l
#SBATCH --job-name=clamp_gt
#SBATCH --output=mlp_gt/clamp_gt_%A_%a.log
#SBATCH --error=mlp_gt/clamp_gt_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-3

# Re-run mediocre mlp_gt GT cases with the clamp(min=1e-4) scale fix.
# Mediocre = datasets where old formula showed flat or inconsistent GT reg effect.
# Task 0: minesweeper  / polynormer (was: -0.25 at lambda=0.4)
# Task 1: minesweeper  / sgformer   (was: flat across all lambdas)
# Task 2: amazon-ratings / polynormer (was: inconsistent, -0.46 to +0.84)
# Task 3: amazon-ratings / sgformer   (was: flat, marginal gains)
#
# Results go to mlp_gt/results_full_clamp.
# Must be submitted from GTs_baselines/ directory.

set -e

DEVICE=0
RUNS=5
RESULT_DIR="mlp_gt/results_full_clamp"

echo "=========================================================="
echo "Clamp-GT Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p "$RESULT_DIR"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

POLYNORMER_HPARAMS="--hidden_channels 128 --local_layers 2 --global_layers 2 --num_heads 4 \
    --lr 0.01 --weight_decay 5e-4 --dropout 0.3 --epochs 1000"

SGFORMER_HPARAMS="--hidden_channels 256 --layers 3 --num_heads 1 \
    --lr 0.01 --weight_decay 5e-4 --dropout 0.3 \
    --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight 0.5 \
    --epochs 1000"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    echo "Dataset: minesweeper | polynormer"
    ./mlp_gt/sweep_mlp_gt.sh \
        "python main.py --model polynormer --dataset minesweeper \
            $POLYNORMER_HPARAMS \
            --metric rocauc \
            --runs $RUNS --device $DEVICE" \
        "$RESULT_DIR"
    ;;
  1)
    echo "Dataset: minesweeper | sgformer"
    ./mlp_gt/sweep_mlp_gt.sh \
        "python main.py --model sgformer --dataset minesweeper \
            $SGFORMER_HPARAMS \
            --metric rocauc \
            --runs $RUNS --device $DEVICE" \
        "$RESULT_DIR"
    ;;
  2)
    echo "Dataset: amazon-ratings | polynormer"
    ./mlp_gt/sweep_mlp_gt.sh \
        "python main.py --model polynormer --dataset amazon-ratings \
            $POLYNORMER_HPARAMS \
            --runs $RUNS --device $DEVICE" \
        "$RESULT_DIR"
    ;;
  3)
    echo "Dataset: amazon-ratings | sgformer"
    ./mlp_gt/sweep_mlp_gt.sh \
        "python main.py --model sgformer --dataset amazon-ratings \
            $SGFORMER_HPARAMS \
            --runs $RUNS --device $DEVICE" \
        "$RESULT_DIR"
    ;;
esac

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "=========================================================="
