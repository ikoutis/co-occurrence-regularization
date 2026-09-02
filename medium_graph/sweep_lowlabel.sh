#!/bin/bash

# Low-label condition sweep (E3.1) for one (dataset, model, budget) cell.
# Conditions are drawn from: baseline, count (train-train edge counting),
# mlp (features-only estimator), oracle (all-label upper bound, LEAKY —
# diagnostic only), shuffle (placebo). Lambda grid is extended upward: at
# small label budgets the CE term is weak and the optimal prior weight can
# exceed the full-supervision grid.
#
# Requeue-resumable via condition stamps (same mechanism as
# sweep_conditions.sh). Delete $RESULT_DIR/.done to force a re-run.
#
# Usage: ./sweep_lowlabel.sh "BASE_CMD" RESULT_DIR "CONDITIONS"
#   BASE_CMD   : quoted python command INCLUDING the budget flags
#                (--label_num_per_class N --resample_split_per_run  or
#                 --train_per_class N) and --budget_tag N
#   RESULT_DIR : where runs_*.csv go (default results/lowlabel)
#   CONDITIONS : space-separated subset of:
#                baseline count mlp oracle shuffle homophily
#                transfer transfer_shuffle   (need $COOC_FILE in the env;
#                                             see TRANSFER_README.md)

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 \"BASE_CMD\" [RESULT_DIR] [\"CONDITIONS\"]"
    exit 1
fi

BASE_CMD="$1"
RESULT_DIR="${2:-results/lowlabel}"
CONDITIONS="${3:-baseline count mlp oracle}"
LAMBDAS="0.01 0.05 0.1 0.2 0.4 0.8 1.6"

echo "=========================================================="
echo "Low-label sweep: $BASE_CMD"
echo "result_dir=$RESULT_DIR conditions=[$CONDITIONS]"
echo "=========================================================="

STAMP_DIR="$RESULT_DIR/.done"
mkdir -p "$STAMP_DIR"

run_condition () {
    local cmd="$1"
    local stamp
    # stamp key ignores --model_dir (contains the SLURM job id) so a
    # resubmitted job still resumes completed conditions
    stamp="$STAMP_DIR/$(printf '%s' "$cmd" | sed 's/--model_dir [^ ]*//' | md5sum | cut -d' ' -f1)"
    if [ -f "$stamp" ]; then
        echo "    (already completed — requeue resume, skipping)"
        return 0
    fi
    eval "$cmd"
    touch "$stamp"
}

for cond in $CONDITIONS; do
    case "$cond" in
      baseline)
        echo "--- baseline ---"
        run_condition "$BASE_CMD --paired_seeds --result_dir $RESULT_DIR"
        ;;
      count)
        for l in $LAMBDAS; do
            echo "--- count_reg | lambda=$l ---"
            run_condition "$BASE_CMD --paired_seeds --use_reg --count_reg \
                --lambda_val $l --result_dir $RESULT_DIR"
        done
        ;;
      mlp)
        for l in $LAMBDAS; do
            echo "--- mlp_reg | lambda=$l ---"
            run_condition "$BASE_CMD --paired_seeds --use_reg --mlp_reg --mlp_epochs 500 \
                --lambda_val $l --result_dir $RESULT_DIR"
        done
        ;;
      oracle)
        for l in $LAMBDAS; do
            echo "--- oracle_reg | lambda=$l ---"
            run_condition "$BASE_CMD --paired_seeds --use_reg --oracle_reg \
                --lambda_val $l --result_dir $RESULT_DIR"
        done
        ;;
      shuffle)
        for l in $LAMBDAS; do
            echo "--- mlp_reg/shuffle placebo | lambda=$l ---"
            run_condition "$BASE_CMD --paired_seeds --use_reg --mlp_reg --mlp_epochs 500 \
                --lambda_val $l --penalty_transform shuffle --result_dir $RESULT_DIR"
        done
        ;;
      homophily)
        for l in $LAMBDAS; do
            echo "--- mlp_reg/homophily control | lambda=$l ---"
            run_condition "$BASE_CMD --paired_seeds --use_reg --mlp_reg --mlp_epochs 500 \
                --lambda_val $l --penalty_transform homophily --result_dir $RESULT_DIR"
        done
        ;;
      transfer|transfer_shuffle)
        if [ -z "$COOC_FILE" ] || [ ! -f "$COOC_FILE" ]; then
            echo "condition $cond needs COOC_FILE (got '$COOC_FILE')"; exit 1
        fi
        TR=""; [ "$cond" = "transfer_shuffle" ] && TR="--penalty_transform shuffle"
        for l in $LAMBDAS; do
            echo "--- transfer prior $COOC_FILE $TR | lambda=$l ---"
            run_condition "$BASE_CMD --paired_seeds --use_reg --cooc_file $COOC_FILE \
                --lambda_val $l $TR --result_dir $RESULT_DIR"
        done
        ;;
      *)
        echo "unknown condition: $cond"; exit 1
        ;;
    esac
done

echo "Low-label sweep completed."
