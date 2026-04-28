#!/bin/bash

# Oracle regularization sweep.
# Penalty matrix is computed from TRUE LABELS (upper-bound experiment).
# Frozen for the entire training. Same per-step normalization.
# Usage: ./sweep_oracle_gnn.sh "python main.py --gnn gcn --dataset ... [HYPERPARAMS]"

if [ -z "$1" ]; then
    echo "Please provide the base python command in quotes."
    exit 1
fi

BASE_CMD="$1"
RESULT_DIR="results/oracle_gnn"

echo "=========================================================="
echo "Starting oracle_gnn Sweep for:"
echo "$BASE_CMD"
echo "=========================================================="

echo "Running Baseline (no reg)..."
eval "$BASE_CMD --result_dir $RESULT_DIR"

for lambda_val in 0.01 0.1 0.2 0.4
do
    echo ""
    echo "Running with Oracle Regularization (lambda=$lambda_val)..."
    eval "$BASE_CMD --use_reg --lambda_val $lambda_val --oracle_reg --result_dir $RESULT_DIR"
done

echo ""
echo "Oracle sweep completed!"
