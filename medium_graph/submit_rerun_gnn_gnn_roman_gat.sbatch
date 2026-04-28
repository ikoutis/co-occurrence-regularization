#!/bin/bash -l
#SBATCH --job-name=gnn_gnn_rerun
#SBATCH --output=results/gnn_gnn_rerun_%j.log
#SBATCH --error=results/gnn_gnn_rerun_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=48:00:00

# Rerun roman-empire / GAT for gnn_gnn sweep.
# Previous run (task 8) was preempted mid-sweep leaving a partial result file
# with duplicate baseline rows. Clear it and start fresh.

set -e

DEVICE=0
RUNS=5
RD=results/gnn_gnn

echo "=========================================================="
echo "GNN-GNN rerun: roman-empire / GAT"
echo "Job: $SLURM_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

# Remove stale partial result before rerunning
rm -f "$RD/roman-empire/MPNN_gat.csv"

./sweep_gnn_gnn.sh "python main.py --gnn gat --dataset roman-empire \
    --hidden_channels 512 --epochs 1000 --lr 0.001 --runs $RUNS \
    --local_layers 10 --weight_decay 5e-4 --dropout 0.3 \
    --bn --res --device $DEVICE"

echo "=========================================================="
echo "Rerun completed at: $(date)"
echo "=========================================================="
