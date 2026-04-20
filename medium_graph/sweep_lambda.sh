#!/bin/bash

# A utility script to sweep lambda values over an optimal baseline configuration.
# Usage: ./sweep_lambda.sh "python main.py --dataset ... [YOUR BEST HYPERPARAMS]"

if [ -z "$1" ]; then
    echo "Please provide the base python command in quotes."
    echo "Example: ./sweep_lambda.sh \"python main.py --dataset roman-empire --gnn gcn --hidden_channels 512\""
    exit 1
fi

BASE_COMMAND="$1"

echo "=========================================================="
echo "Starting Lambda Sweep for Baseline Command:"
echo "$BASE_COMMAND"
echo "=========================================================="

# Run the baseline (lambda=0, no regularization)
echo "Running Baseline (REG: False)..."
eval $BASE_COMMAND

# Run the lambda sweep
for lambda_val in 0.1 0.5 1.0 2.0 5.0
do
    echo ""
    echo "Running with Dynamic Regularization (lambda=$lambda_val)..."
    eval "$BASE_COMMAND --use_reg --lambda_val $lambda_val"
done

echo ""
echo "Sweep completed! Check the results CSV to see the optimal lambda."
