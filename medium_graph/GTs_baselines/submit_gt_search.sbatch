#!/bin/bash -l
#SBATCH --job-name=gt_search
#SBATCH --output=gt_search_%A_%a.log
#SBATCH --error=gt_search_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-14
#SBATCH --requeue

# Bounded hyperparameter search for the (model, dataset) pairs with NO
# published config:
#   - Polynormer: cora, citeseer, pubmed, chameleon, squirrel
#     (official repo covers only the 10 Polynormer-paper datasets)
#   - SGFormer: the 10 datasets outside its paper's medium-graph set
#     (incl. squirrel, whose official config uses --method difformer and is
#     not reproducible with this SGFormer implementation)
#
# Search spaces bracket the published configs of each model. 3 runs per
# config; select by MEAN VALIDATION accuracy with analyze_gt_search.py, then
# confirm the winner with 10 runs before using it as a baseline.
#
# Must be submitted from GTs_baselines/.

set -e

DEVICE=0
RUNS=3
RESULT_DIR="gt_search/results"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$RESULT_DIR"
nvidia-smi

COMBOS=(
  "polynormer cora"
  "polynormer citeseer"
  "polynormer pubmed"
  "polynormer chameleon"
  "polynormer squirrel"
  "sgformer amazon-computer"
  "sgformer amazon-photo"
  "sgformer coauthor-cs"
  "sgformer coauthor-physics"
  "sgformer wikics"
  "sgformer roman-empire"
  "sgformer amazon-ratings"
  "sgformer minesweeper"
  "sgformer questions"
  "sgformer squirrel"
)

read -r MODEL DATASET <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
echo "Searching: $MODEL / $DATASET"

# dataset-specific protocol flags (mirror run_gnn.sh so GT and GNN results
# stay split-comparable)
case "$DATASET" in
  cora|citeseer|pubmed)
    SPLIT_FLAGS="--rand_split_class --valid_num 500 --test_num 1000 --seed 123"
    ;;
  minesweeper|questions)
    SPLIT_FLAGS="--metric rocauc"
    ;;
  *)
    SPLIT_FLAGS=""
    ;;
esac

case "$DATASET" in
  roman-empire|amazon-ratings) EPOCHS=2500 ;;
  chameleon|squirrel)          EPOCHS=500  ;;
  *)                           EPOCHS=1000 ;;
esac

if [ "$MODEL" == "polynormer" ]; then
  # bracket the official configs: deep local, shallow global, in_dropout 0.2
  for local_layers in 3 5 7; do
  for global_layers in 1 2; do
  for dropout in 0.3 0.5 0.7; do
    python main.py --model polynormer --dataset $DATASET \
      --hidden_channels 64 --num_heads 8 --lr 0.001 --weight_decay 5e-4 \
      --local_layers $local_layers --global_layers $global_layers \
      --dropout $dropout --in_dropout 0.2 \
      --local_epochs 100 --global_epochs $EPOCHS --save_model \
      --model_dir models/gt_search_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
      --runs $RUNS --device $DEVICE --result_dir $RESULT_DIR $SPLIT_FLAGS
  done; done; done
else
  # bracket the official SGFormer configs: shallow GCN backbone, 1 trans layer
  case "$DATASET" in
    roman-empire|amazon-ratings|minesweeper|questions) BN_FLAG="--use_bn" ;;
    *) BN_FLAG="" ;;
  esac
  for layers in 2 3; do
  for hidden in 64 256; do
  for dropout in 0.3 0.5 0.7; do
  for graph_weight in 0.5 0.8; do
    python main.py --model sgformer --dataset $DATASET \
      --lr 0.001 --layers $layers --tr_layers 1 --hidden_channels $hidden \
      --weight_decay 5e-4 --dropout $dropout \
      --use_graph --graph_weight $graph_weight --use_residual --alpha 0.5 \
      --num_heads 1 $BN_FLAG --epochs $EPOCHS \
      --runs $RUNS --device $DEVICE --result_dir $RESULT_DIR $SPLIT_FLAGS
  done; done; done; done
fi

rm -rf "models/gt_search_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Search done: $MODEL / $DATASET. Select with: python analyze_gt_search.py"
