#!/bin/bash -l
#SBATCH --job-name=gnn_gnn
#SBATCH --output=results/gnn_gnn_%A_%a.log
#SBATCH --error=results/gnn_gnn_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-14
#SBATCH --requeue

set -e

# Dynamic GNN-GNN regularization sweep on 5 candidate datasets:
#   - amazon-ratings, roman-empire, minesweeper (heterophilic)
#   - cora, citeseer (homophilic, GNN >> MLP)
# Each: 3 GNNs (gcn, sage, gat) x 5 lambdas (0, 0.01, 0.1, 0.2, 0.4) = 15 tasks
#
# Penalty matrix updated every 50 epochs from GNN's own outputs.
# Per-step normalization (loss.detach() / reg_loss.detach().abs()).

DEVICE=0
RUNS=5
RD=results/gnn_gnn

echo "=========================================================="
echo "GNN-GNN Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p results

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

run_gnn() {
    ./sweep_gnn_gnn.sh "python main.py --gnn $1 --dataset $2 --hidden_channels $3 \
        --epochs $4 --lr $5 --runs $RUNS --local_layers $6 --weight_decay $7 \
        --dropout $8 --device $DEVICE ${@:9}"
}

case "$SLURM_ARRAY_TASK_ID" in
  # ---------------- amazon-ratings ----------------
  0) run_gnn gcn  amazon-ratings 512 1000 0.001 4 5e-4 0.5 --bn --res ;;
  1) run_gnn sage amazon-ratings 512 1000 0.001 9 5e-4 0.5 --bn --res ;;
  2) run_gnn gat  amazon-ratings 512 1000 0.001 4 5e-4 0.5 --bn --res ;;

  # ---------------- minesweeper (binary -> rocauc) ----------------
  3) run_gnn gcn  minesweeper 64 1000 0.01 12 5e-4 0.2 --bn --res --metric rocauc ;;
  4) run_gnn sage minesweeper 64 1000 0.01 15 5e-4 0.2 --bn --res --metric rocauc ;;
  5) run_gnn gat  minesweeper 64 1000 0.01 15 5e-4 0.2 --bn --res --metric rocauc ;;

  # ---------------- roman-empire ----------------
  6) run_gnn gcn  roman-empire 512 1000 0.001 9  5e-4 0.5 --bn --res ;;
  7) run_gnn sage roman-empire 256 1000 0.001 9  5e-4 0.3 --bn ;;
  8) run_gnn gat  roman-empire 512 1000 0.001 10 5e-4 0.3 --bn --res ;;

  # ---------------- cora ----------------
  9)  run_gnn gcn  cora 512 1000 0.001 3 5e-4 0.7 \
         --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  10) run_gnn sage cora 256 1000 0.001 3 5e-4 0.7 \
         --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  11) run_gnn gat  cora 512 1000 0.001 3 5e-4 0.2 \
         --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --res ;;

  # ---------------- citeseer ----------------
  12) run_gnn gcn  citeseer 512 1000 0.001 2 0.01 0.5 \
         --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  13) run_gnn sage citeseer 512 1000 0.001 3 0.01 0.2 \
         --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  14) run_gnn gat  citeseer 256 1000 0.001 3 0.01 0.5 \
         --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --res ;;

  *)
    echo "ERROR: unknown task $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "=========================================================="
