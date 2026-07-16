#!/bin/bash -l
#SBATCH --job-name=gnn_unified
#SBATCH --output=results/gnn_unified_%A_%a.log
#SBATCH --error=results/gnn_unified_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-83
#SBATCH --requeue

# Unified GNN re-run under the fixed protocol (EXPERIMENT_PLAN E2.1 + E2.3):
# every (dataset, model) line of run_gnn.sh (the tunedGNN-optimized configs,
# VERBATIM — including --metric rocauc for minesweeper/questions and the
# --pre_linear/--bn/--res flags whose absence produced the degenerate
# roman-empire/GCN baseline), each swept over:
#   baseline + mlp_reg x lambda{0.01,0.05,0.1,0.2,0.4} + oracle x lambda grid
# with 10 paired runs. The 8 placebo pairs are EXCLUDED here — they run in
# submit_placebo.sbatch with the full transform set (its results/placebo CSVs
# contain the same baseline/mlp-none/oracle conditions and are merged at
# analysis time).
#
# Each run_gnn.sh line is split into 2 condition groups so 2500-epoch
# datasets stay inside the 48h limit: task_id = line_idx * 2 + group
#   group 0: baseline + mlp/none x lambda grid (6 configs)
#   group 1: oracle x lambda grid              (5 configs)
# Pairing across groups holds via --paired_seeds.
# Must be submitted from medium_graph/.

set -e

DEVICE=0
RUNS=10
RESULT_DIR="results/unified"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"
mkdir -p results "$RESULT_DIR"
nvidia-smi

mapfile -t LINES < <(grep '^python main.py' run_gnn.sh)
echo "run_gnn.sh contains ${#LINES[@]} commands"
LINE_IDX=$((SLURM_ARRAY_TASK_ID / 2))
GROUP=$((SLURM_ARRAY_TASK_ID % 2))
if [ "$LINE_IDX" -ge "${#LINES[@]}" ]; then
    echo "Line idx beyond command count — nothing to do."; exit 0
fi

CMD="${LINES[$LINE_IDX]}"

# pairs covered by the placebo job (full transform set) — skip here
PLACEBO_PAIRS=(
  "gcn cora" "gat cora" "gat citeseer" "gat squirrel" "gcn squirrel"
  "sage amazon-ratings" "gcn amazon-ratings" "gat roman-empire"
)
for pair in "${PLACEBO_PAIRS[@]}"; do
    read -r G D <<< "$pair"
    if [[ "$CMD" == *"--gnn $G "* && "$CMD" == *"--dataset $D "* ]]; then
        echo "($G, $D) is covered by submit_placebo.sbatch — skipping."; exit 0
    fi
done

# run_gnn.sh lines use --device $1; substitute, then override runs (argparse
# takes the last occurrence of a repeated flag)
CMD="${CMD//\$1/$DEVICE}"
CMD="$CMD --runs $RUNS"

echo "$CMD"
if [ "$GROUP" = "0" ]; then
    ./sweep_conditions.sh "$CMD" "$RESULT_DIR" "none" 0 1
else
    ./sweep_conditions.sh "$CMD" "$RESULT_DIR" ""     1 0
fi

echo "Task $SLURM_ARRAY_TASK_ID done."
