#!/bin/bash -l
#SBATCH --job-name=mlp_gt_missing
#SBATCH --output=mlp_gt/mlp_gt_missing_%A_%a.log
#SBATCH --error=mlp_gt/mlp_gt_missing_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --requeue

# Rerun missing GT tasks from the full sweep:
#   0: polynormer / chameleon
#   1: polynormer / squirrel
#   2: sgformer   / chameleon
#   3: sgformer   / squirrel
#   4: nodeformer / chameleon  (requires torch_sparse with CUDA)
#   5: nodeformer / squirrel   (requires torch_sparse with CUDA)
#
# Plus a separate set of tasks for nodeformer on all 14 datasets
# can be submitted once torch_sparse CUDA is confirmed working.
#
# Must be submitted from GTs_baselines/ directory.

set -e

DEVICE=0
RUNS=5
RESULT_DIR="mlp_gt/results_full"

echo "=========================================================="
echo "MLP-GT Missing Tasks | Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p "$RESULT_DIR"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    MODEL="polynormer"
    DATASET="chameleon"
    HPARAMS="--hidden_channels 128 --local_layers 2 --global_layers 2 --num_heads 4 --lr 0.01 --weight_decay 5e-4 --dropout 0.3 --epochs 1000"
    SPLIT_FLAGS=""
    ;;
  1)
    MODEL="polynormer"
    DATASET="squirrel"
    HPARAMS="--hidden_channels 128 --local_layers 2 --global_layers 2 --num_heads 4 --lr 0.01 --weight_decay 5e-4 --dropout 0.3 --epochs 1000"
    SPLIT_FLAGS=""
    ;;
  2)
    MODEL="sgformer"
    DATASET="chameleon"
    HPARAMS="--hidden_channels 256 --layers 3 --num_heads 1 --lr 0.01 --weight_decay 5e-4 --dropout 0.3 --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight 0.5 --epochs 1000"
    SPLIT_FLAGS=""
    ;;
  3)
    MODEL="sgformer"
    DATASET="squirrel"
    HPARAMS="--hidden_channels 256 --layers 3 --num_heads 1 --lr 0.01 --weight_decay 5e-4 --dropout 0.3 --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight 0.5 --epochs 1000"
    SPLIT_FLAGS=""
    ;;
  4)
    MODEL="nodeformer"
    DATASET="chameleon"
    HPARAMS="--hidden_channels 128 --global_layers 2 --num_heads 4 --lr 0.01 --weight_decay 5e-4 --dropout 0.3 --use_bn --epochs 1000"
    SPLIT_FLAGS=""
    ;;
  5)
    MODEL="nodeformer"
    DATASET="squirrel"
    HPARAMS="--hidden_channels 128 --global_layers 2 --num_heads 4 --lr 0.01 --weight_decay 5e-4 --dropout 0.3 --use_bn --epochs 1000"
    SPLIT_FLAGS=""
    ;;
  *)
    echo "ERROR: unknown task $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "Model: $MODEL | Dataset: $DATASET"

./mlp_gt/sweep_mlp_gt.sh \
    "python main.py --model $MODEL --dataset $DATASET \
        $HPARAMS \
        --runs $RUNS --device $DEVICE \
        $SPLIT_FLAGS" \
    "$RESULT_DIR"

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID ($MODEL / $DATASET) completed at: $(date)"
echo "=========================================================="
