#!/bin/bash -l
#SBATCH --job-name=mlp_gnn_rerun
#SBATCH --output=results/mlp_gnn_rerun_%A_%a.log
#SBATCH --error=results/mlp_gnn_rerun_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-9
#SBATCH --requeue

set -e

# Re-run for tasks missing from the main sweep:
#   0-3: squirrel    (MLP, gcn, sage, gat)
#   4-7: chameleon   (MLP, gcn, sage, gat)
#   8:   coauthor-physics / sage
#   9:   questions / gcn (full re-run; deletes stale 4-row file first)

DEVICE=0
RUNS=5
RD=results/mlp_gnn

echo "=========================================================="
echo "MLP-GNN Rerun Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p results

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

run_mlp() {
    python main.py --model MLP --dataset "$1" --hidden_channels "$2" --local_layers "$3" \
        --dropout "$4" --lr "$5" --weight_decay "$6" --epochs "$7" --runs $RUNS \
        --device $DEVICE --result_dir $RD ${@:8}
}

run_gnn() {
    ./sweep_mlp_gnn.sh "python main.py --gnn $1 --dataset $2 --hidden_channels $3 \
        --epochs $4 --lr $5 --runs $RUNS --local_layers $6 --weight_decay $7 \
        --dropout $8 --device $DEVICE ${@:9}"
}

case "$SLURM_ARRAY_TASK_ID" in
  # ---------------- squirrel ----------------
  0) run_mlp squirrel 256 4 0.7 0.01 5e-4 1000 ;;
  1) run_gnn gcn  squirrel 256 1000 0.01  4 5e-4 0.7 --ln --res ;;
  2) run_gnn sage squirrel 256 1000 0.01  3 5e-4 0.7 --ln --res ;;
  3) run_gnn gat  squirrel 512 1000 0.005 7 5e-4 0.5 --ln --res ;;

  # ---------------- chameleon ----------------
  4) run_mlp chameleon 512 4 0.2 0.005 5e-4 1000 ;;
  5) run_gnn gcn  chameleon 512 1000 0.005 5 5e-4 0.2 ;;
  6) run_gnn sage chameleon 256 1000 0.01  4 5e-4 0.7 --ln --res ;;
  7) run_gnn gat  chameleon 256 1000 0.01  2 5e-4 0.7 --ln --res ;;

  # ---------------- coauthor-physics / sage ----------------
  8) run_gnn sage coauthor-physics 64 1500 0.001 2 5e-4 0.7 --bn --res ;;

  # ---------------- questions / gcn (delete stale 4-row file, full re-run) ----------------
  9)
    rm -f $RD/questions/MPNN_gcn.csv
    run_gnn gcn questions 512 1000 3e-5 10 5e-4 0.3 --res --metric rocauc
    ;;

  *)
    echo "ERROR: unknown task $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "=========================================================="
echo "Rerun task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "=========================================================="
