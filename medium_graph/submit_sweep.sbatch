#!/bin/bash
#SBATCH --job-name=gnn_sweep
#SBATCH --output=results/gnn_sweep_%j.log
#SBATCH --error=results/gnn_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a800:1     # Request exactly one A800 GPU
#SBATCH --time=48:00:00       # Allow up to 48 hours for the massive sweep
#SBATCH --partition=gpu       # Adjust this if Wulver uses a different partition name like 'gpu-a800' or 'gpu_long'

echo "=========================================================="
echo "Starting GNN Sweep on NJIT Wulver Cluster"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "=========================================================="

# 1. Load any necessary modules (Uncomment and adjust if Wulver requires 'module load miniconda3' or 'module load cuda')
# module load miniconda3

# 2. Activate your conda environment (Replace 'gnn_reg' with whatever you named it on Wulver, e.g., 'co_occurrence')
source ~/.bashrc  # Ensures conda is initialized in this subshell
conda activate gnn_reg

# 3. Navigate to the code directory (Assuming you submit from within medium_graph, but it's safe to enforce it)
cd $SLURM_SUBMIT_DIR

# 4. Kick off the main Bash script
# Slurm manages CUDA_VISIBLE_DEVICES automatically, so your assigned GPU will always be index 0
bash run_gnn_sweep.sh 0

echo "=========================================================="
echo "Sweep Completed at: $(date)"
echo "=========================================================="
