#!/bin/bash -l
#SBATCH --job-name=nodeformer_full
#SBATCH --output=mlp_gt/nodeformer_full_%A_%a.log
#SBATCH --error=mlp_gt/nodeformer_full_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-13
#SBATCH --requeue

# NodeFormer sweep across all 14 datasets.
# PREREQUISITE: torch_sparse must be installed with CUDA support.
# Verify before submitting:
#   python -c "import torch_sparse; t = torch_sparse.SparseTensor(
#       row=torch.zeros(1,dtype=torch.long).cuda(),
#       col=torch.zeros(1,dtype=torch.long).cuda(),
#       sparse_sizes=(1,1)); print('CUDA OK')"
#
# Must be submitted from GTs_baselines/ directory.

set -e

DEVICE=0
RUNS=5
RESULT_DIR="mlp_gt/results_full"

DATASETS=(cora citeseer pubmed amazon-ratings minesweeper questions roman-empire amazon-computer amazon-photo coauthor-cs coauthor-physics wikics chameleon squirrel)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

HPARAMS="--hidden_channels 128 --global_layers 2 --num_heads 4 \
    --lr 0.01 --weight_decay 5e-4 --dropout 0.3 --use_bn --epochs 1000"

case "$DATASET" in
  cora|citeseer|pubmed|amazon-computer|amazon-photo|coauthor-cs|coauthor-physics)
    SPLIT_FLAGS="--rand_split_class --valid_num 500 --test_num 1000 --seed 123"
    ;;
  minesweeper|questions)
    SPLIT_FLAGS="--metric rocauc"
    ;;
  *)
    SPLIT_FLAGS=""
    ;;
esac

echo "=========================================================="
echo "NodeFormer Full Sweep | Task $SLURM_ARRAY_TASK_ID | Dataset: $DATASET"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p "$RESULT_DIR"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi

# Ensure torch_sparse CUDA kernels work on this node (SparseTensor creation
# is not enough — must test .csr() which calls the ind2ptr CUDA kernel).
python -c "
import torch, torch_sparse
row = torch.zeros(1,dtype=torch.long).cuda()
col = torch.zeros(1,dtype=torch.long).cuda()
t = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=(1,1))
t.csr()
print('torch_sparse CUDA OK')
" 2>/dev/null || {
    echo 'torch_sparse CUDA kernels not working — reinstalling with CUDA support...'
    module load cuda/11.6 2>/dev/null || module load cuda 2>/dev/null || true
    nvcc --version 2>/dev/null || echo 'nvcc not found, build may be CPU-only'
    pip uninstall torch_sparse torch_scatter -y
    pip install torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html --no-cache-dir
    python -c "
import torch, torch_sparse
row = torch.zeros(1,dtype=torch.long).cuda()
col = torch.zeros(1,dtype=torch.long).cuda()
t = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=(1,1))
t.csr()
print('torch_sparse CUDA OK after reinstall')
"
}

python -c "import torch; import torch_sparse; print(torch.__version__, torch_sparse.__version__, torch.cuda.is_available())"

./mlp_gt/sweep_mlp_gt.sh \
    "python main.py --model nodeformer --dataset $DATASET \
        $HPARAMS \
        --runs $RUNS --device $DEVICE \
        $SPLIT_FLAGS" \
    "$RESULT_DIR"

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID (nodeformer / $DATASET) completed at: $(date)"
echo "=========================================================="
