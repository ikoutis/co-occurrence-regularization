#!/bin/bash

# Canonical condition sweep for one (dataset, model) pair, all paired-seeded:
#   baseline (optional) + mlp_reg x {lambda grid} x {transforms} + oracle x {lambda grid}
#
# Lambda grid includes 0.01 because several original wins (e.g. squirrel)
# peaked there; 0.05 fills the old 0.01->0.1 gap.
#
# Usage: ./sweep_conditions.sh "BASE_CMD" RESULT_DIR "TRANSFORMS" ORACLE RUN_BASELINE MLP_EPOCHS
#   BASE_CMD     : quoted python command (hyperparams, no reg/result flags)
#   RESULT_DIR   : where results + runs_*.csv go        (default results/unified)
#   TRANSFORMS   : space-separated subset of: none shuffle homophily (default "none")
#   ORACLE       : 1 to add oracle_reg conditions        (default 1)
#   RUN_BASELINE : 1 to run the no-reg baseline first    (default 1; set 0 when
#                  the baseline already exists in RESULT_DIR with the same seeds)
#   MLP_EPOCHS   : pre-training epochs for the penalty MLP (default 500;
#                  10/50 give deliberately degraded penalties for the
#                  penalty-quality curve, EXPERIMENT_PLAN E4.3)

set -e  # a crashed condition must fail the SLURM task, not be silently skipped

if [ -z "$1" ]; then
    echo "Usage: $0 \"BASE_CMD\" [RESULT_DIR] [\"TRANSFORMS\"] [ORACLE] [RUN_BASELINE]"
    exit 1
fi

BASE_CMD="$1"
RESULT_DIR="${2:-results/unified}"
# ${3-none} (not ${3:-none}): an explicit empty string means "no mlp
# conditions" (e.g. an oracle-only group); only an OMITTED arg defaults
TRANSFORMS="${3-none}"
ORACLE="${4:-1}"
RUN_BASELINE="${5:-1}"
MLP_EPOCHS="${6:-500}"
LAMBDAS="0.01 0.05 0.1 0.2 0.4"

echo "=========================================================="
echo "Condition sweep: $BASE_CMD"
echo "result_dir=$RESULT_DIR transforms=[$TRANSFORMS] oracle=$ORACLE baseline=$RUN_BASELINE"
echo "=========================================================="

if [ "$RUN_BASELINE" = "1" ]; then
    echo "--- baseline (no reg) ---"
    eval "$BASE_CMD --paired_seeds --result_dir $RESULT_DIR"
fi

for transform in $TRANSFORMS; do
    for lambda_val in $LAMBDAS; do
        echo "--- mlp_reg | transform=$transform | lambda=$lambda_val | mlp_epochs=$MLP_EPOCHS ---"
        eval "$BASE_CMD --paired_seeds --use_reg --mlp_reg --mlp_epochs $MLP_EPOCHS \
              --lambda_val $lambda_val --penalty_transform $transform \
              --result_dir $RESULT_DIR"
    done
done

if [ "$ORACLE" = "1" ]; then
    for lambda_val in $LAMBDAS; do
        echo "--- oracle_reg | lambda=$lambda_val ---"
        eval "$BASE_CMD --paired_seeds --use_reg --oracle_reg \
              --lambda_val $lambda_val --result_dir $RESULT_DIR"
    done
fi

echo "Condition sweep completed."
