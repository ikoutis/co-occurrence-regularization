#!/bin/bash -l
#SBATCH --job-name=arxiv_oracle
#SBATCH --output=results/arxiv_oracle_%A_%a.log
#SBATCH --error=results/arxiv_oracle_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-1

# Oracle (true-label) co-occurrence regularization on ogbn-arxiv.
# Task 0: GCN  (5 layers, hidden 512)
# Task 1: SAGE (4 layers, hidden 256)
#
# Must be submitted from large_graph/ directory.

set -e

DEVICE=0
RUNS=5
RESULT_DIR="results/arxiv_oracle"

echo "=========================================================="
echo "ogbn-arxiv ORACLE sweep | Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p "$RESULT_DIR/ogbn-arxiv"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    BASE_CMD="python main-arxiv.py --dataset ogbn-arxiv \
        --hidden_channels 512 --local_layers 5 \
        --lr 0.0005 --weight_decay 5e-4 --dropout 0.5 \
        --bn --res --runs $RUNS --device $DEVICE --epochs 2000"
    ;;
  1)
    BASE_CMD="python main-arxiv.py --dataset ogbn-arxiv \
        --hidden_channels 256 --local_layers 4 \
        --lr 0.0005 --weight_decay 5e-4 --dropout 0.5 \
        --bn --res --sage --runs $RUNS --device $DEVICE --epochs 2000"
    ;;
esac

./sweep_arxiv_oracle.sh "$BASE_CMD" "$RESULT_DIR"

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "=========================================================="
