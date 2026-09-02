#!/bin/bash -l
#SBATCH --job-name=lowlabel_ctrl
#SBATCH --output=results/lowlabel_ctrl_%A_%a.log
#SBATCH --error=results/lowlabel_ctrl_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-127
#SBATCH --requeue

# Low-label CONTROLS at budgets 10 and 20 — the two budgets where the
# leakage-free estimators showed their only significant gains (cora/GCN,
# cora/Poly, wikics/GCN, wikics/Poly, amazon-computer/Poly) but where the
# main grid (submit_lowlabel.sbatch) ran no placebo and no matched-budget
# validation check. Fills exactly those holes:
#
#   group 0: mlp/shuffle placebo      -> results/lowlabel     (joins main grid)
#   group 1: mlp/homophily control    -> results/lowlabel     (diagonal only:
#                                        is it homophily or class-pair structure?)
#   group 2: matched-val baseline     -> results/lowlabel_mv  (--valid_per_class)
#   group 3: matched-val mlp          -> results/lowlabel_mv
#
#   8 datasets x {GCN, Polynormer} x budgets {10, 20} x 4 groups = 128 tasks
#   task_id = ((dataset_idx * 2 + model_idx) * 2 + budget_idx) * 4 + group
#
# Everything else (configs, seeds, split resampling, lambda grid, stamps) is
# identical to submit_lowlabel.sbatch, so the new rows pair with the existing
# baselines run-for-run. Requeue-resumable. Must be submitted from medium_graph/.

set -e

DEVICE=0
RUNS=10

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"
mkdir -p results results/lowlabel results/lowlabel_mv
nvidia-smi

DATASETS=(cora citeseer pubmed amazon-computer coauthor-cs wikics amazon-ratings roman-empire)
MODELS=(gcn polynormer)
BUDGETS=(10 20)

IDX=$SLURM_ARRAY_TASK_ID
GROUP=$((IDX % 4)); IDX=$((IDX / 4))
BUDGET=${BUDGETS[$((IDX % 2))]}; IDX=$((IDX / 2))
MODEL=${MODELS[$((IDX % 2))]}; IDX=$((IDX / 2))
DATASET=${DATASETS[$IDX]}

case $GROUP in
  0) SWEEP_CONDS="shuffle";   RESULT_DIR="results/lowlabel";    MV="" ;;
  1) SWEEP_CONDS="homophily"; RESULT_DIR="results/lowlabel";    MV="" ;;
  2) SWEEP_CONDS="baseline";  RESULT_DIR="results/lowlabel_mv"; MV="--valid_per_class $BUDGET" ;;
  3) SWEEP_CONDS="mlp";       RESULT_DIR="results/lowlabel_mv"; MV="--valid_per_class $BUDGET" ;;
esac

echo "Control cell: $DATASET / $MODEL / budget $BUDGET per class / $SWEEP_CONDS -> $RESULT_DIR"

case "$DATASET" in
  cora|citeseer|pubmed)
    BUDGET_FLAGS="--label_num_per_class $BUDGET --resample_split_per_run $MV --budget_tag $BUDGET"
    ;;
  *)
    BUDGET_FLAGS="--train_per_class $BUDGET $MV --budget_tag $BUDGET"
    ;;
esac

if [ "$MODEL" = "gcn" ]; then
    CMD=$(grep -m1 -- "--gnn gcn --dataset $DATASET " run_gnn.sh || true)
    if [ -z "$CMD" ]; then echo "ERROR: no run_gnn.sh line for $DATASET"; exit 1; fi
    CMD="${CMD//\$1/$DEVICE}"
    CMD="$CMD --runs $RUNS"
else
    MODEL_DIR="models/lowlabel_ctrl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    cd GTs_baselines
    case "$DATASET" in
      cora|citeseer|pubmed)
        CMD=$(python gt_confirm_cmd.py --model polynormer --dataset $DATASET \
              --runs $RUNS --device $DEVICE --model_dir $MODEL_DIR \
              --out_result_dir ../$RESULT_DIR)
        ;;
      *)
        CMD=$(grep -m1 -- "--dataset $DATASET " polynormer_tuned.sh || true)
        if [ -z "$CMD" ]; then echo "ERROR: no polynormer_tuned.sh line for $DATASET"; exit 1; fi
        EXTRA="--model_dir $MODEL_DIR"
        CMD=$(eval echo "$CMD")
        ;;
    esac
    RESULT_DIR="../$RESULT_DIR"
fi

CMD="$CMD $BUDGET_FLAGS"
echo "$CMD"

if [ "$MODEL" = "gcn" ]; then
    ./sweep_lowlabel.sh "$CMD" "$RESULT_DIR" "$SWEEP_CONDS"
else
    ../sweep_lowlabel.sh "$CMD" "$RESULT_DIR" "$SWEEP_CONDS"
    rm -rf "$MODEL_DIR"
fi

echo "Task $SLURM_ARRAY_TASK_ID done."
