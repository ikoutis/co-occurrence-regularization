#!/bin/bash -l
#SBATCH --job-name=mlp_gt_full
#SBATCH --output=mlp_gt/mlp_gt_full_%A_%a.log
#SBATCH --error=mlp_gt/mlp_gt_full_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-41
#SBATCH --requeue

# Full MLP-GT sweep: 14 datasets x 3 linear-attention GT models.
# Layout: task_id = model_idx * 14 + dataset_idx
#   model_idx: 0=polynormer, 1=nodeformer, 2=sgformer
#   dataset_idx: 0-13 (see DATASETS array below)
#
# Results go to mlp_gt/results_full/ to stay separate from the cora pilot.
# The key question: do GTs without local MP (polynormer, nodeformer) benefit
# more from co-occurrence regularization than GTs with local GNN backbone
# (sgformer) or standard MPNNs (gcn/gat/sage)?
#
# Must be submitted from GTs_baselines/ directory.

set -e

DEVICE=0
RUNS=5
RESULT_DIR="mlp_gt/results_full"

echo "=========================================================="
echo "MLP-GT Full Sweep | Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p "$RESULT_DIR"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

DATASETS=(cora citeseer pubmed amazon-ratings minesweeper questions roman-empire amazon-computer amazon-photo coauthor-cs coauthor-physics wikics chameleon squirrel)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 14))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % 14))
DATASET=${DATASETS[$DATASET_IDX]}

case $MODEL_IDX in
  0)
    MODEL="polynormer"
    MODEL_HPARAMS="--hidden_channels 128 --local_layers 2 --global_layers 2 --num_heads 4 \
        --lr 0.01 --weight_decay 5e-4 --dropout 0.3 --epochs 1000"
    ;;
  1)
    MODEL="nodeformer"
    MODEL_HPARAMS="--hidden_channels 128 --global_layers 2 --num_heads 4 \
        --lr 0.01 --weight_decay 5e-4 --dropout 0.3 --use_bn --epochs 1000"
    ;;
  2)
    MODEL="sgformer"
    MODEL_HPARAMS="--hidden_channels 256 --layers 3 --num_heads 1 \
        --lr 0.01 --weight_decay 5e-4 --dropout 0.3 \
        --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight 0.5 \
        --epochs 1000"
    ;;
  *)
    echo "ERROR: unknown model_idx $MODEL_IDX"
    exit 1
    ;;
esac

# rand_split_class for citation/co-purchase networks; rocauc for binary tasks;
# fixed splits (no extra flags) for everything else.
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

echo "Model: $MODEL | Dataset: $DATASET"
echo "Split flags: $SPLIT_FLAGS"

./mlp_gt/sweep_mlp_gt.sh \
    "python main.py --model $MODEL --dataset $DATASET \
        $MODEL_HPARAMS \
        --runs $RUNS --device $DEVICE \
        $SPLIT_FLAGS" \
    "$RESULT_DIR"

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID ($MODEL / $DATASET) completed at: $(date)"
echo "=========================================================="
