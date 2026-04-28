#!/bin/bash

# Sweeps only small lambda values (no baseline run) using MLP-based regularization.
# Usage: ./sweep_lambda_mlp_small.sh "python main.py --dataset ... [YOUR BEST HYPERPARAMS]"

if [ -z "$1" ]; then
    echo "Please provide the base python command in quotes."
    exit 1
fi

BASE_COMMAND="$1"

echo "=========================================================="
echo "Starting Small Lambda MLP-Reg Sweep for:"
echo "$BASE_COMMAND"
echo "=========================================================="

for lambda_val in 1e-4 0.001 0.005
do
    echo ""
    echo "Running with Pre-trained MLP Regularization (lambda=$lambda_val)..."
    eval "$BASE_COMMAND --use_reg --lambda_val $lambda_val --mlp_reg --mlp_epochs 500"
done

echo ""
echo "Small lambda sweep completed!"
