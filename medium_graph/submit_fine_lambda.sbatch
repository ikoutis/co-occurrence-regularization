#!/bin/bash -l
#SBATCH --job-name=mlp_gnn_fine
#SBATCH --output=results/mlp_gnn_fine_%A_%a.log
#SBATCH --error=results/mlp_gnn_fine_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --requeue

set -e

# Fill-in lambda sweep for squirrel/chameleon (small heterophilic graphs).
# Adds lambda = 0.005, 0.05 to existing {0, 0.01, 0.1, 0.2, 0.4} grid.
#
#   0: squirrel  / gcn
#   1: squirrel  / sage
#   2: squirrel  / gat
#   3: chameleon / gcn
#   4: chameleon / sage
#   5: chameleon / gat

DEVICE=0
RUNS=5
RD=results/mlp_gnn

echo "=========================================================="
echo "MLP-GNN Fine Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p results

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

run_gnn_fine() {
    ./sweep_mlp_gnn_fine.sh "python main.py --gnn $1 --dataset $2 --hidden_channels $3 \
        --epochs $4 --lr $5 --runs $RUNS --local_layers $6 --weight_decay $7 \
        --dropout $8 --device $DEVICE ${@:9}"
}

case "$SLURM_ARRAY_TASK_ID" in
  # ---------------- squirrel ----------------
  0) run_gnn_fine gcn  squirrel 256 1000 0.01  4 5e-4 0.7 --ln --res ;;
  1) run_gnn_fine sage squirrel 256 1000 0.01  3 5e-4 0.7 --ln --res ;;
  2) run_gnn_fine gat  squirrel 512 1000 0.005 7 5e-4 0.5 --ln --res ;;

  # ---------------- chameleon ----------------
  3) run_gnn_fine gcn  chameleon 512 1000 0.005 5 5e-4 0.2 ;;
  4) run_gnn_fine sage chameleon 256 1000 0.01  4 5e-4 0.7 --ln --res ;;
  5) run_gnn_fine gat  chameleon 256 1000 0.01  2 5e-4 0.7 --ln --res ;;

  *)
    echo "ERROR: unknown task $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "=========================================================="
echo "Fine task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "=========================================================="
