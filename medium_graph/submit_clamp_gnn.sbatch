#!/bin/bash -l
#SBATCH --job-name=clamp_gnn
#SBATCH --output=results/clamp_gnn_%A_%a.log
#SBATCH --error=results/clamp_gnn_%A_%a.err
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

# Re-run mediocre mlp_gnn GCN cases with the clamp(min=1e-4) scale fix.
# Mediocre = datasets where old formula showed flat or negative reg effect.
# Task 0: pubmed      (was: -0.14 at lambda=0.4)
# Task 1: minesweeper (was: +0.06 at lambda=0.4, near flat)
# Task 2: coauthor-cs (was: -0.10 at lambda=0.4)
# Task 3: squirrel    (was: -2.67 at lambda=0.4)
#
# Must be submitted from medium_graph/ directory.

set -e

DEVICE=0
RUNS=5
RD="results/mlp_gnn_clamp"

echo "=========================================================="
echo "Clamp-GNN Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p "$RD"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

sweep() {
    local BASE_CMD="$1"
    echo "Running GCN baseline (no reg)..."
    eval "$BASE_CMD --result_dir $RD"
    for lambda_val in 0.01 0.1 0.2 0.4; do
        echo "Running MLP_REG lambda=$lambda_val..."
        eval "$BASE_CMD --use_reg --mlp_reg --lambda_val $lambda_val --mlp_epochs 500 --result_dir $RD"
    done
}

case "$SLURM_ARRAY_TASK_ID" in
  0)
    echo "Dataset: pubmed | GCN"
    sweep "python main.py --gnn gcn --dataset pubmed \
        --hidden_channels 256 --local_layers 2 --epochs 1000 \
        --lr 0.005 --weight_decay 5e-4 --dropout 0.7 \
        --rand_split_class --valid_num 500 --test_num 1000 --seed 123 \
        --runs $RUNS --device $DEVICE"
    ;;
  1)
    echo "Dataset: minesweeper | GCN"
    sweep "python main.py --gnn gcn --dataset minesweeper \
        --hidden_channels 64 --local_layers 12 --epochs 1000 \
        --lr 0.01 --weight_decay 5e-4 --dropout 0.2 \
        --bn --res --metric rocauc \
        --runs $RUNS --device $DEVICE"
    ;;
  2)
    echo "Dataset: coauthor-cs | GCN"
    sweep "python main.py --gnn gcn --dataset coauthor-cs \
        --hidden_channels 512 --local_layers 2 --epochs 1500 \
        --lr 0.001 --weight_decay 5e-4 --dropout 0.3 \
        --ln --res \
        --runs $RUNS --device $DEVICE"
    ;;
  3)
    echo "Dataset: squirrel | GCN"
    sweep "python main.py --gnn gcn --dataset squirrel \
        --hidden_channels 256 --local_layers 4 --epochs 1000 \
        --lr 0.01 --weight_decay 5e-4 --dropout 0.7 \
        --ln --res \
        --runs $RUNS --device $DEVICE"
    ;;
esac

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "=========================================================="
