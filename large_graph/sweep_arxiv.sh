#!/bin/bash
# Sweep co-occurrence regularization on ogbn-arxiv.
# Usage: ./sweep_arxiv.sh "python main-arxiv.py [BASE_ARGS]" [result_dir]
# Must be called from large_graph/ directory.

if [ -z "$1" ]; then
    echo "Usage: $0 \"python main-arxiv.py [BASE_ARGS]\" [result_dir]"
    exit 1
fi

BASE_CMD="$1"
RESULT_DIR="${2:-results}"

echo "=========================================================="
echo "Starting arxiv sweep"
echo "CMD: $BASE_CMD"
echo "Result dir: $RESULT_DIR"
echo "=========================================================="

echo "Running baseline (no reg)..."
eval "$BASE_CMD --save_result --result_dir $RESULT_DIR"

for lambda_val in 0.01 0.1 0.2 0.4; do
    echo ""
    echo "Running with MLP_REG lambda=$lambda_val..."
    eval "$BASE_CMD --use_reg --mlp_reg --lambda_val $lambda_val --mlp_epochs 500 --save_result --result_dir $RESULT_DIR"
done

echo ""
echo "Sweep completed!"
