#!/bin/bash -l
#SBATCH --job-name=mlp_gt
#SBATCH --output=mlp_gt/mlp_gt_%A_%a.log
#SBATCH --error=mlp_gt/mlp_gt_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --requeue

set -e

# Pilot: MLP-derived co-occurrence regularization applied to Graph Transformers.
# Dataset: cora (2.7k nodes, best mlp_gnn wins: +1.04 for GAT, +0.54 for GCN)
# Models: SGFormer (0), Polynormer (1), NodeFormer (2)  [linear-attention only]
# Lambda: {0, 0.01, 0.1, 0.2, 0.4}  -- 5 runs each
# Must be submitted from GTs_baselines/ directory.

DEVICE=0
RUNS=5

echo "=========================================================="
echo "MLP-GT Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p mlp_gt/results

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

sweep() {
    ./mlp_gt/sweep_mlp_gt.sh "$1"
}

case "$SLURM_ARRAY_TASK_ID" in
  # SGFormer / cora
  # Linear complexity attention; fast and memory-efficient.
  0)
    sweep "python main.py --model sgformer --dataset cora \
        --hidden_channels 256 --layers 3 --num_heads 1 \
        --lr 0.01 --weight_decay 5e-4 --dropout 0.3 \
        --epochs 1000 --runs $RUNS --device $DEVICE \
        --use_graph --use_act --alpha 0.5 --use_residual --use_bn \
        --graph_weight 0.5 \
        --rand_split_class --valid_num 500 --test_num 1000 --seed 123"
    ;;

  # Polynormer / cora
  # Local + equivariant global attention; proven on citation graphs.
  1)
    sweep "python main.py --model polynormer --dataset cora \
        --hidden_channels 128 --local_layers 2 --global_layers 2 --num_heads 4 \
        --lr 0.01 --weight_decay 5e-4 --dropout 0.3 \
        --epochs 1000 --runs $RUNS --device $DEVICE \
        --rand_split_class --valid_num 500 --test_num 1000 --seed 123"
    ;;

  # NodeFormer / cora
  # Kernel-approximated all-pair attention (linear complexity).
  2)
    sweep "python main.py --model nodeformer --dataset cora \
        --hidden_channels 128 --global_layers 2 --num_heads 4 \
        --lr 0.01 --weight_decay 5e-4 --dropout 0.3 \
        --epochs 1000 --runs $RUNS --device $DEVICE \
        --use_bn \
        --rand_split_class --valid_num 500 --test_num 1000 --seed 123"
    ;;

  *)
    echo "ERROR: unknown task $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "=========================================================="
