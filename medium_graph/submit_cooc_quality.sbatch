#!/bin/bash -l
#SBATCH --job-name=cooc_quality
#SBATCH --output=results/cooc_quality_%A_%a.log
#SBATCH --error=results/cooc_quality_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3

# Q1 diagnostic: does co-occurrence regularization improve co-occurrence statistics?
# Trains GCN (baseline + mlp_reg sweep) and measures:
#   - Test accuracy
#   - Normalized Frobenius distance ||C_pred - C_oracle||_F / ||C_oracle||_F
#
# Datasets chosen to cover key cases:
#   Task 0: cora          — GCN acc gain +0.54, expect Fro to improve
#   Task 1: coauthor-cs   — GCN acc gain ~0,   does Fro still improve?
#   Task 2: amazon-ratings — GCN acc gain +0.72, heterophilic case
#   Task 3: amazon-computer — GCN acc gain -0.14, slightly negative
#
# Must be submitted from medium_graph/ directory.

set -e

DEVICE=0
RUNS=3

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "=========================================================="
echo "cooc_quality Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p results/cooc_quality

case "$SLURM_ARRAY_TASK_ID" in
  0)
    echo "Dataset: cora"
    python measure_cooc_quality.py \
        --dataset cora \
        --hidden_channels 512 --local_layers 3 --epochs 1000 \
        --lr 0.001 --weight_decay 5e-4 --dropout 0.7 \
        --rand_split_class --valid_num 500 --test_num 1000 --seed 123 \
        --runs $RUNS --device $DEVICE
    ;;
  1)
    echo "Dataset: coauthor-cs"
    python measure_cooc_quality.py \
        --dataset coauthor-cs \
        --hidden_channels 512 --local_layers 2 --epochs 1500 \
        --lr 0.001 --weight_decay 5e-4 --dropout 0.3 \
        --ln --res \
        --rand_split_class --valid_num 500 --test_num 1000 --seed 123 \
        --runs $RUNS --device $DEVICE
    ;;
  2)
    echo "Dataset: amazon-ratings"
    python measure_cooc_quality.py \
        --dataset amazon-ratings \
        --hidden_channels 512 --local_layers 4 --epochs 1000 \
        --lr 0.001 --weight_decay 5e-4 --dropout 0.5 \
        --bn --res \
        --runs $RUNS --device $DEVICE
    ;;
  3)
    echo "Dataset: amazon-computer"
    python measure_cooc_quality.py \
        --dataset amazon-computer \
        --hidden_channels 512 --local_layers 3 --epochs 1000 \
        --lr 0.001 --weight_decay 5e-5 --dropout 0.5 \
        --ln \
        --rand_split_class --valid_num 500 --test_num 1000 --seed 123 \
        --runs $RUNS --device $DEVICE
    ;;
esac

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "=========================================================="
