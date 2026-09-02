#!/bin/bash -l
#SBATCH --job-name=transfer
#SBATCH --output=results/transfer_%A_%a.log
#SBATCH --error=results/transfer_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-47
#SBATCH --requeue

# E4 — a LEGITIMATE source for the co-occurrence matrix. See TRANSFER_README.md.
#
# ogbn-arxiv, temporal split at YEAR: the prior is counted on the labeled
# past (papers < YEAR); train/valid/test are drawn from papers >= YEAR at a
# small per-class budget. GCN only (tunedGNN arxiv config, frozen).
#
# 16 (source_mode, condition) cells x 3 budgets = 48 tasks:
#   keep   : baseline count mlp oracle transfer transfer_shuffle transfer_m100 transfer_m20
#   drop   : baseline count mlp oracle transfer transfer_shuffle
#   labels : baseline transfer
#   task_id = budget_idx * 16 + cell_idx
#
# BEFORE submitting, build the priors once on the login node (downloads the
# dataset if needed, ~1 min):
#   python make_cooc_prior.py --year_split 2018 --out priors/arxiv_lt2018_all.pt  [--data_dir ...]
#   python make_cooc_prior.py --year_split 2018 --source_per_class 100 --out priors/arxiv_lt2018_m100.pt
#   python make_cooc_prior.py --year_split 2018 --source_per_class 20  --out priors/arxiv_lt2018_m20.pt
# (tasks rebuild a missing prior themselves, but 48 concurrent first-time
# dataset downloads is not something to rely on)
#
# Requeue-resumable (stamps under results/transfer_<mode>/.done).
# Must be submitted from medium_graph/.

set -e

DEVICE=0
RUNS=10
YEAR=2018
DATA_DIR="./data/"
# reuse the copy the large_graph runs already downloaded, if present
[ -d "../large_graph/data/ogb/ogbn_arxiv" ] && DATA_DIR="../large_graph/data/"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"
mkdir -p results priors
nvidia-smi

BUDGETS=(5 20 50)
CELLS=(
  "keep baseline" "keep count" "keep mlp" "keep oracle"
  "keep transfer" "keep transfer_shuffle" "keep transfer_m100" "keep transfer_m20"
  "drop baseline" "drop count" "drop mlp" "drop oracle"
  "drop transfer" "drop transfer_shuffle"
  "labels baseline" "labels transfer"
)

IDX=$SLURM_ARRAY_TASK_ID
CELL=${CELLS[$((IDX % 16))]}
BUDGET=${BUDGETS[$((IDX / 16))]}
MODE=${CELL%% *}
COND=${CELL#* }

# transfer_m100 / transfer_m20: prior from a subsampled past
PRIOR_TAG="all"
case "$COND" in
  transfer_m100) PRIOR_TAG="m100"; COND="transfer" ;;
  transfer_m20)  PRIOR_TAG="m20";  COND="transfer" ;;
esac
RESULT_DIR="results/transfer_${MODE}"
[ "$PRIOR_TAG" != "all" ] && RESULT_DIR="results/transfer_${MODE}_${PRIOR_TAG}"
mkdir -p "$RESULT_DIR"

echo "Transfer cell: mode=$MODE cond=$COND prior=$PRIOR_TAG budget=$BUDGET/class -> $RESULT_DIR"

export COOC_FILE="priors/arxiv_lt${YEAR}_${PRIOR_TAG}.pt"
if [ "$COND" = "transfer" ] || [ "$COND" = "transfer_shuffle" ]; then
    if [ ! -f "$COOC_FILE" ]; then
        case "$PRIOR_TAG" in
          all)  SPC=0 ;;
          m100) SPC=100 ;;
          m20)  SPC=20 ;;
        esac
        python make_cooc_prior.py --dataset ogbn-arxiv --data_dir "$DATA_DIR" \
            --year_split $YEAR --source_per_class $SPC --out "$COOC_FILE"
    fi
fi

# tunedGNN ogbn-arxiv GCN config (large_graph/arxiv.sh), frozen; the target
# pool holds ~78k papers, so valid/test are sized for 40 classes
CMD="python main.py --gnn gcn --dataset ogbn-arxiv --data_dir $DATA_DIR \
  --hidden_channels 512 --epochs 2000 --lr 0.0005 --local_layers 5 --bn --res \
  --device $DEVICE --runs $RUNS \
  --rand_split_class --label_num_per_class $BUDGET --resample_split_per_run \
  --valid_num 2000 --test_num 20000 \
  --year_split $YEAR --source_mode $MODE --budget_tag $BUDGET"
echo "$CMD"

./sweep_lowlabel.sh "$CMD" "$RESULT_DIR" "$COND"

echo "Task $SLURM_ARRAY_TASK_ID done."
