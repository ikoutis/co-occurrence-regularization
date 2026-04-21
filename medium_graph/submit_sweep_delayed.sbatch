#!/bin/bash -l
#SBATCH --job-name=gnn_delayed_sweep
#SBATCH --output=results/gnn_delayed_sweep_%j.log
#SBATCH --error=results/gnn_delayed_sweep_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=48:00:00
#SBATCH --requeue

set -e

echo "=========================================================="
echo "Starting Delayed GNN Sweep on NJIT Wulver Cluster"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "=========================================================="

mkdir -p results

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

bash run_gnn_sweep_delayed.sh 0

echo "=========================================================="
echo "Sweep Completed at: $(date)"
echo "=========================================================="
