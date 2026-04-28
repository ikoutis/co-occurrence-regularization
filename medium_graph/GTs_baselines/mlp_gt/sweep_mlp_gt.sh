#!/bin/bash

# MLP-GT sweep: static MLP-derived penalty applied to a Graph Transformer.
# Runs GT baseline (no reg) then lambda = 0.01, 0.1, 0.2, 0.4.
# Must be called from GTs_baselines/ directory.
# Usage: ./mlp_gt/sweep_mlp_gt.sh "python main.py --model polynormer --dataset cora [HYPERPARAMS]"

if [ -z "$1" ]; then
    echo "Please provide the base python command in quotes."
    exit 1
fi

BASE_CMD="$1"
RESULT_DIR="mlp_gt/results"

echo "=========================================================="
echo "Starting mlp_gt GT Sweep for:"
echo "$BASE_CMD"
echo "=========================================================="

echo "Running GT Baseline (no reg)..."
eval "$BASE_CMD --result_dir $RESULT_DIR"

for lambda_val in 0.01 0.1 0.2 0.4
do
    echo ""
    echo "Running with MLP-GT Regularization (lambda=$lambda_val)..."
    eval "$BASE_CMD --use_reg --lambda_val $lambda_val --mlp_reg --mlp_epochs 500 --result_dir $RESULT_DIR"
done

echo ""
echo "GT sweep completed!"
