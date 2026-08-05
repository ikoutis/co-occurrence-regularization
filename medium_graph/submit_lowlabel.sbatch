#!/bin/bash -l
#SBATCH --job-name=lowlabel
#SBATCH --output=results/lowlabel_%A_%a.log
#SBATCH --error=results/lowlabel_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-399
#SBATCH --requeue

# Low-label experiment (E3.1): accuracy vs label budget for co-occurrence
# regularization, under the full paired protocol. See LOWLABEL_README.md.
#
#   8 datasets x {GCN, Polynormer} x budgets {2,5,10,20,50}/class x 5 groups
#     group 0: baseline                       (1 config)
#     group 1: count x 7 lambdas              (7 configs)
#     group 2: mlp x 7 lambdas                (7 configs)
#     group 3: oracle x 7 lambdas             (7 configs)
#     group 4: shuffle placebo x 7 lambdas    (budgets 2 and 5 only)
#   task_id = ((dataset_idx * 2 + model_idx) * 5 + budget_idx) * 5 + group
#   (groups capped at 7 trainings x 10 runs so even the heaviest
#   2700-epoch Polynormer cells stay inside 48h; requeue-resume covers
#   any residual overrun)
#
# Hyperparameters are the FULL-SUPERVISION tuned configs, frozen (run_gnn.sh
# for GCN; polynormer_tuned.sh / search winners for Polynormer) — the paired
# delta is config-fair, and the budget-50/20 cells join the full-supervision
# anchor. Budgets:
#   cora/citeseer/pubmed: --label_num_per_class N --resample_split_per_run
#     (a fresh label sample per run, seeded; val/test protocol unchanged)
#   all others (fixed splits): --train_per_class N (per-run subsample of the
#     fixed train split; val/test untouched)
#
# Must be submitted from medium_graph/.

set -e

DEVICE=0
RUNS=10
RESULT_DIR="results/lowlabel"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"
mkdir -p results "$RESULT_DIR"
nvidia-smi

DATASETS=(cora citeseer pubmed amazon-computer coauthor-cs wikics amazon-ratings roman-empire)
MODELS=(gcn polynormer)
BUDGETS=(2 5 10 20 50)

IDX=$SLURM_ARRAY_TASK_ID
GROUP=$((IDX % 5)); IDX=$((IDX / 5))
BUDGET=${BUDGETS[$((IDX % 5))]}; IDX=$((IDX / 5))
MODEL=${MODELS[$((IDX % 2))]}; IDX=$((IDX / 2))
DATASET=${DATASETS[$IDX]}

echo "Cell: $DATASET / $MODEL / budget $BUDGET per class / group $GROUP"

# shuffle placebo only at the two smallest budgets
if [ "$GROUP" = "4" ] && [ "$BUDGET" != "2" ] && [ "$BUDGET" != "5" ]; then
    echo "shuffle group only runs at budgets 2 and 5 — nothing to do."; exit 0
fi

# budget flags per dataset family
case "$DATASET" in
  cora|citeseer|pubmed)
    BUDGET_FLAGS="--label_num_per_class $BUDGET --resample_split_per_run --budget_tag $BUDGET"
    ;;
  *)
    BUDGET_FLAGS="--train_per_class $BUDGET --budget_tag $BUDGET"
    ;;
esac

if [ "$MODEL" = "gcn" ]; then
    CMD=$(grep -m1 -- "--gnn gcn --dataset $DATASET " run_gnn.sh || true)
    if [ -z "$CMD" ]; then echo "ERROR: no run_gnn.sh line for $DATASET"; exit 1; fi
    CMD="${CMD//\$1/$DEVICE}"
    CMD="$CMD --runs $RUNS"
else
    MODEL_DIR="models/lowlabel_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    cd GTs_baselines
    case "$DATASET" in
      cora|citeseer|pubmed)
        # search-winner configs (validation-selected from gt_search results)
        CMD=$(python gt_confirm_cmd.py --model polynormer --dataset $DATASET \
              --runs $RUNS --device $DEVICE --model_dir $MODEL_DIR \
              --out_result_dir ../$RESULT_DIR)
        ;;
      *)
        # published configs (lines reference $RUNS/$DEVICE/$EXTRA)
        CMD=$(grep -m1 -- "--dataset $DATASET " polynormer_tuned.sh || true)
        if [ -z "$CMD" ]; then echo "ERROR: no polynormer_tuned.sh line for $DATASET"; exit 1; fi
        EXTRA="--model_dir $MODEL_DIR"
        CMD=$(eval echo "$CMD")
        ;;
    esac
    RESULT_DIR="../results/lowlabel"
fi

CMD="$CMD $BUDGET_FLAGS"
echo "$CMD"

case $GROUP in
  0) SWEEP_CONDS="baseline" ;;
  1) SWEEP_CONDS="count" ;;
  2) SWEEP_CONDS="mlp" ;;
  3) SWEEP_CONDS="oracle" ;;
  4) SWEEP_CONDS="shuffle" ;;
esac

if [ "$MODEL" = "gcn" ]; then
    ./sweep_lowlabel.sh "$CMD" "$RESULT_DIR" "$SWEEP_CONDS"
else
    ../sweep_lowlabel.sh "$CMD" "$RESULT_DIR" "$SWEEP_CONDS"
    rm -rf "$MODEL_DIR"
fi

echo "Task $SLURM_ARRAY_TASK_ID done."
