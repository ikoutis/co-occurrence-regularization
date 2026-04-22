#!/bin/bash

# A utility script to sweep lambda values over an optimal baseline configuration.
# Generates the regularization penalty matrix statically using a pre-trained MLP!
# Usage: ./sweep_lambda_mlp.sh "python main.py --dataset ... [YOUR BEST HYPERPARAMS]"

if [ -z "$1" ]; then
    echo "Please provide the base python command in quotes."
    exit 1
fi

BASE_COMMAND="$1"

echo "=========================================================="
echo "Starting MLP-Based Inference Sweep for Baseline Command:"
echo "$BASE_COMMAND"
echo "=========================================================="

# Run the baseline (lambda=0, no regularization)
echo "Running Baseline (REG: False)..."
eval $BASE_COMMAND

# Run the lambda sweep
for lambda_val in 0.01 0.05 0.1 0.5 1.0
do
    echo ""
    echo "Running with Pre-trained MLP Regularization (lambda=$lambda_val)..."
    eval "$BASE_COMMAND --use_reg --lambda_val $lambda_val --mlp_reg --mlp_epochs 500"
done

echo ""
echo "MLP Inference Sweep completed!"
