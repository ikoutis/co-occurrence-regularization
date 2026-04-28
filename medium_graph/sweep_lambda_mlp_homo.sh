#!/bin/bash

# Lambda sweep for homophilic datasets — runs only the most informative values.
# Usage: ./sweep_lambda_mlp_homo.sh "python main.py --dataset ... [YOUR BEST HYPERPARAMS]"

if [ -z "$1" ]; then
    echo "Please provide the base python command in quotes."
    exit 1
fi

BASE_COMMAND="$1"

echo "=========================================================="
echo "Starting Homophilic MLP-Reg Sweep for:"
echo "$BASE_COMMAND"
echo "=========================================================="

echo "Running Baseline (REG: False)..."
eval $BASE_COMMAND

for lambda_val in 0.01 0.1 1.0
do
    echo ""
    echo "Running with Pre-trained MLP Regularization (lambda=$lambda_val)..."
    eval "$BASE_COMMAND --use_reg --lambda_val $lambda_val --mlp_reg --mlp_epochs 500"
done

echo ""
echo "Homophilic sweep completed!"
