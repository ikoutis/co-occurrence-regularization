#!/bin/bash -l
#SBATCH --job-name=transfer_f
#SBATCH --output=results/transfer_f_%A_%a.log
#SBATCH --error=results/transfer_f_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=12:00:00
#SBATCH --array=0-335
#SBATCH --requeue

# Per-LAMBDA version of submit_transfer.sbatch (same cells, same commands,
# same stamps) — one (cell, budget, lambda) per task, ~1 h each, so that the
# qos=low preemption cycle can no longer kill a 7-lambda keep-mode task
# before it completes. Already-finished conditions are skipped by the
# stamps, so this can be submitted on top of a partially completed
# submit_transfer.sbatch run.
#
#   task_id = (budget_idx * 16 + cell_idx) * 7 + lambda_idx
#   baseline cells run only at lambda_idx 0 (the others exit immediately).
#
# Priors must exist in priors/ (see submit_transfer.sbatch header).
# Must be submitted from medium_graph/.

set -e

DEVICE=0
RUNS=10
YEAR=2018
DATA_DIR="./data/"
[ -d "../large_graph/data/ogb/ogbn_arxiv" ] && DATA_DIR="../large_graph/data/"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"
mkdir -p results priors
nvidia-smi

BUDGETS=(5 20 50)
LAMBDAS=(0.01 0.05 0.1 0.2 0.4 0.8 1.6)
CELLS=(
  "keep baseline" "keep count" "keep mlp" "keep oracle"
  "keep transfer" "keep transfer_shuffle" "keep transfer_m100" "keep transfer_m20"
  "drop baseline" "drop count" "drop mlp" "drop oracle"
  "drop transfer" "drop transfer_shuffle"
  "labels baseline" "labels transfer"
)

IDX=$SLURM_ARRAY_TASK_ID
LI=$((IDX % 7)); IDX=$((IDX / 7))
CELL=${CELLS[$((IDX % 16))]}
BUDGET=${BUDGETS[$((IDX / 16))]}
MODE=${CELL%% *}
COND=${CELL#* }
export LAMBDAS_OVERRIDE="${LAMBDAS[$LI]}"

if [ "$COND" = "baseline" ] && [ "$LI" != "0" ]; then
    echo "baseline has no lambda — nothing to do."; exit 0
fi

PRIOR_TAG="all"
case "$COND" in
  transfer_m100) PRIOR_TAG="m100"; COND="transfer" ;;
  transfer_m20)  PRIOR_TAG="m20";  COND="transfer" ;;
esac
RESULT_DIR="results/transfer_${MODE}"
[ "$PRIOR_TAG" != "all" ] && RESULT_DIR="results/transfer_${MODE}_${PRIOR_TAG}"
mkdir -p "$RESULT_DIR"

echo "Transfer cell: mode=$MODE cond=$COND prior=$PRIOR_TAG budget=$BUDGET/class lambda=$LAMBDAS_OVERRIDE -> $RESULT_DIR"

export COOC_FILE="priors/arxiv_lt${YEAR}_${PRIOR_TAG}.pt"
if [ "$COND" = "transfer" ] || [ "$COND" = "transfer_shuffle" ]; then
    if [ ! -f "$COOC_FILE" ]; then
        case "$PRIOR_TAG" in all) SPC=0 ;; m100) SPC=100 ;; m20) SPC=20 ;; esac
        python make_cooc_prior.py --dataset ogbn-arxiv --data_dir "$DATA_DIR" \
            --year_split $YEAR --source_per_class $SPC --out "$COOC_FILE"
    fi
fi

CMD="python main.py --gnn gcn --dataset ogbn-arxiv --data_dir $DATA_DIR \
  --hidden_channels 512 --epochs 2000 --lr 0.0005 --local_layers 5 --bn --res \
  --device $DEVICE --runs $RUNS \
  --rand_split_class --label_num_per_class $BUDGET --resample_split_per_run \
  --valid_num 2000 --test_num 20000 \
  --year_split $YEAR --source_mode $MODE --budget_tag $BUDGET"
echo "$CMD"

./sweep_lowlabel.sh "$CMD" "$RESULT_DIR" "$COND"

echo "Task $SLURM_ARRAY_TASK_ID done."
