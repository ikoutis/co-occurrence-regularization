#!/bin/bash -l
#SBATCH --job-name=roman_gat_rerun
#SBATCH --output=results/roman_gat_rerun_%A_%a.log
#SBATCH --error=results/roman_gat_rerun_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-2

# Rerun roman-empire / GAT for all three penalty variants with correct config.
# Previous runs used res=False for a 10-layer GAT → model collapsed to ~18%
# (near random on 18 classes). Correct config requires --res to avoid
# over-smoothing. Confirmed: rerun baseline = 88.12%.
#
# Task 0: gnn_gnn   — complete the sweep (missing lambda=0.4 + fresh full run)
# Task 1: mlp_gnn   — rerun with --res (previous 18% result invalid)
# Task 2: oracle    — rerun with --res (previous 18% result invalid)
#
# Must be submitted from medium_graph/ directory.

set -e

DEVICE=0
RUNS=5

BASE_CMD="python main.py --gnn gat --dataset roman-empire \
    --hidden_channels 512 --epochs 1000 --lr 0.001 --runs $RUNS \
    --local_layers 10 --weight_decay 5e-4 --dropout 0.3 \
    --bn --res --device $DEVICE"

echo "=========================================================="
echo "roman-empire / GAT rerun | Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

case "$SLURM_ARRAY_TASK_ID" in

  # gnn_gnn: clear stale partial file and rerun full sweep
  0)
    rm -f results/gnn_gnn/roman-empire/MPNN_gat.csv
    ./sweep_gnn_gnn.sh "$BASE_CMD"
    ;;

  # mlp_gnn: clear invalid 18% result and rerun
  1)
    rm -f results/mlp_gnn/roman-empire/MPNN_gat.csv
    ./sweep_mlp_gnn.sh "$BASE_CMD"
    ;;

  # oracle: clear invalid 18% result and rerun
  2)
    rm -f results/oracle_gnn/roman-empire/MPNN_gat.csv
    ./sweep_oracle_gnn.sh "$BASE_CMD"
    ;;

  *)
    echo "ERROR: unknown task $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "=========================================================="
