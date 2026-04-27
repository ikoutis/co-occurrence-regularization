#!/bin/bash

# Fine-grained lambda fill-in for squirrel/chameleon.
# Adds lambda = 0.005 and 0.05 to existing CSVs (which already have
# {0, 0.01, 0.1, 0.2, 0.4} from the main sweep).
# Usage: ./sweep_mlp_gnn_fine.sh "python main.py --gnn gcn --dataset ... [HYPERPARAMS]"

if [ -z "$1" ]; then
    echo "Please provide the base python command in quotes."
    exit 1
fi

BASE_CMD="$1"
RESULT_DIR="results/mlp_gnn"

echo "=========================================================="
echo "Starting mlp_gnn FINE Sweep for:"
echo "$BASE_CMD"
echo "=========================================================="

for lambda_val in 0.005 0.05
do
    echo ""
    echo "Running with MLP-GNN Regularization (lambda=$lambda_val)..."
    eval "$BASE_CMD --use_reg --lambda_val $lambda_val --mlp_reg --mlp_epochs 500 --result_dir $RESULT_DIR"
done

echo ""
echo "Fine sweep completed!"
