#!/bin/bash -l
#SBATCH --job-name=ctrl_f
#SBATCH --output=results/lowlabel_ctrl_f_%A_%a.log
#SBATCH --error=results/lowlabel_ctrl_f_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=12:00:00
#SBATCH --array=0-895
#SBATCH --requeue

# Per-LAMBDA version of submit_lowlabel_ctrl.sbatch: identical cells,
# commands and stamps, one lambda per task, so heavy Polynormer cells
# survive the qos=low preemption cycle. Submit only the leftovers, e.g.
#   sbatch --array=$(python ctrl_leftovers.py) submit_lowlabel_ctrl_fine.sbatch
#
#   task_id = ctrl_task_id * 7 + lambda_idx     (ctrl_task_id as in submit_lowlabel_ctrl.sbatch)
#   the matched-val baseline group runs only at lambda_idx 0.
#
# Must be submitted from medium_graph/.

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
LAMBDAS=(0.01 0.05 0.1 0.2 0.4 0.8 1.6)

IDX=$SLURM_ARRAY_TASK_ID
LI=$((IDX % 7)); IDX=$((IDX / 7))
CTRL_ID=$IDX
GROUP=$((IDX % 4)); IDX=$((IDX / 4))
BUDGET=${BUDGETS[$((IDX % 2))]}; IDX=$((IDX / 2))
MODEL=${MODELS[$((IDX % 2))]}; IDX=$((IDX / 2))
DATASET=${DATASETS[$IDX]}
export LAMBDAS_OVERRIDE="${LAMBDAS[$LI]}"

case $GROUP in
  0) SWEEP_CONDS="shuffle";   RESULT_DIR="results/lowlabel";    MV="" ;;
  1) SWEEP_CONDS="homophily"; RESULT_DIR="results/lowlabel";    MV="" ;;
  2) SWEEP_CONDS="baseline";  RESULT_DIR="results/lowlabel_mv"; MV="--valid_per_class $BUDGET" ;;
  3) SWEEP_CONDS="mlp";       RESULT_DIR="results/lowlabel_mv"; MV="--valid_per_class $BUDGET" ;;
esac
if [ "$GROUP" = "2" ] && [ "$LI" != "0" ]; then
    echo "baseline has no lambda — nothing to do."; exit 0
fi

echo "Control cell (ctrl task $CTRL_ID): $DATASET / $MODEL / budget $BUDGET per class / $SWEEP_CONDS lambda=$LAMBDAS_OVERRIDE -> $RESULT_DIR"

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
