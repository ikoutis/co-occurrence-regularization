#!/bin/bash

# Dynamic GNN-based regularization sweep.
# Penalty matrix is updated from the GNN's own predictions every
# reg_update_freq epochs (here: 50). Results go to results/gnn_gnn/.
# Runs GNN baseline (no reg) then lambda = 0.01, 0.1, 0.2, 0.4.
# Usage: ./sweep_gnn_gnn.sh "python main.py --gnn gcn --dataset ... [HYPERPARAMS]"

if [ -z "$1" ]; then
    echo "Please provide the base python command in quotes."
    exit 1
fi

BASE_CMD="$1"
RESULT_DIR="results/gnn_gnn"

echo "=========================================================="
echo "Starting gnn_gnn (dynamic) Sweep for:"
echo "$BASE_CMD"
echo "=========================================================="

echo "Running GNN Baseline (no reg)..."
eval "$BASE_CMD --result_dir $RESULT_DIR"

for lambda_val in 0.01 0.1 0.2 0.4
do
    echo ""
    echo "Running with GNN-GNN Regularization (lambda=$lambda_val, freq=50)..."
    eval "$BASE_CMD --use_reg --lambda_val $lambda_val --reg_update_freq 50 --result_dir $RESULT_DIR"
done

echo ""
echo "GNN-GNN sweep completed!"
