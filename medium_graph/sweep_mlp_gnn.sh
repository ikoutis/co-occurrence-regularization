#!/bin/bash

# GNN sweep for the mlp_gnn regularization method.
# Runs GNN baseline (no reg) then lambda = 0.01, 0.1, 0.2, 0.4.
# MLP baseline is run separately (see submit_sweep_mlp_gnn.sbatch).
# Usage: ./sweep_mlp_gnn.sh "python main.py --gnn gcn --dataset ... [HYPERPARAMS]"

if [ -z "$1" ]; then
    echo "Please provide the base python command in quotes."
    exit 1
fi

BASE_CMD="$1"
RESULT_DIR="results/mlp_gnn"

echo "=========================================================="
echo "Starting mlp_gnn GNN Sweep for:"
echo "$BASE_CMD"
echo "=========================================================="

echo "Running GNN Baseline (no reg)..."
eval "$BASE_CMD --result_dir $RESULT_DIR"

for lambda_val in 0.01 0.1 0.2 0.4
do
    echo ""
    echo "Running with MLP-GNN Regularization (lambda=$lambda_val)..."
    eval "$BASE_CMD --use_reg --lambda_val $lambda_val --mlp_reg --mlp_epochs 500 --result_dir $RESULT_DIR"
done

echo ""
echo "GNN sweep completed!"
