#!/bin/bash -l
#SBATCH --job-name=lowlabel_mv
#SBATCH --output=results/lowlabel_mv_%A_%a.log
#SBATCH --error=results/lowlabel_mv_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-63
#SBATCH --requeue

# Matched-budget validation sensitivity check (the low-label evaluation
# pitfall): at tiny budgets the standard 500-node validation set carries
# more supervision than the training set. This re-runs baseline + mlp at
# budgets {2, 5} with the validation set subsampled to the SAME per-class
# budget (--valid_per_class). If lambda* and the deltas agree with the main
# run (results/lowlabel), the curves are trustworthy; if not, that
# instability is a finding to report.
#
#   8 datasets x {GCN, Polynormer} x budgets {2,5} x groups {baseline, mlp}
#   task_id = ((dataset_idx * 2 + model_idx) * 2 + budget_idx) * 2 + group
#
# Must be submitted from medium_graph/.

set -e

DEVICE=0
RUNS=10
RESULT_DIR="results/lowlabel_mv"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"
mkdir -p results "$RESULT_DIR"
nvidia-smi

DATASETS=(cora citeseer pubmed amazon-computer coauthor-cs wikics amazon-ratings roman-empire)
MODELS=(gcn polynormer)
BUDGETS=(2 5)

IDX=$SLURM_ARRAY_TASK_ID
MVGROUP=$((IDX % 2)); IDX=$((IDX / 2))
BUDGET=${BUDGETS[$((IDX % 2))]}; IDX=$((IDX / 2))
MODEL=${MODELS[$((IDX % 2))]}; IDX=$((IDX / 2))
DATASET=${DATASETS[$IDX]}

echo "Matched-val cell: $DATASET / $MODEL / budget $BUDGET per class / group $MVGROUP"

case "$DATASET" in
  cora|citeseer|pubmed)
    BUDGET_FLAGS="--label_num_per_class $BUDGET --resample_split_per_run --valid_per_class $BUDGET --budget_tag $BUDGET"
    ;;
  *)
    BUDGET_FLAGS="--train_per_class $BUDGET --valid_per_class $BUDGET --budget_tag $BUDGET"
    ;;
esac

if [ "$MODEL" = "gcn" ]; then
    CMD=$(grep -m1 -- "--gnn gcn --dataset $DATASET " run_gnn.sh || true)
    if [ -z "$CMD" ]; then echo "ERROR: no run_gnn.sh line for $DATASET"; exit 1; fi
    CMD="${CMD//\$1/$DEVICE}"
    CMD="$CMD --runs $RUNS"
else
    MODEL_DIR="models/lowlabel_mv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
    RESULT_DIR="../results/lowlabel_mv"
fi

CMD="$CMD $BUDGET_FLAGS"
echo "$CMD"

[ "$MVGROUP" = "0" ] && MVCONDS="baseline" || MVCONDS="mlp"
if [ "$MODEL" = "gcn" ]; then
    ./sweep_lowlabel.sh "$CMD" "$RESULT_DIR" "$MVCONDS"
else
    ../sweep_lowlabel.sh "$CMD" "$RESULT_DIR" "$MVCONDS"
    rm -rf "$MODEL_DIR"
fi

echo "Task $SLURM_ARRAY_TASK_ID done."
