#!/bin/bash

# Placebo sweep: baseline + {true, shuffled, homophily-only} MLP-derived
# penalty, each over the lambda grid, all with paired seeds.
# See PLACEBO_README.md for the design rationale.
# Usage: ./sweep_placebo.sh "python main.py --gnn gcn --dataset ... [HYPERPARAMS]"

if [ -z "$1" ]; then
    echo "Please provide the base python command in quotes."
    exit 1
fi

BASE_CMD="$1"
RESULT_DIR="results/placebo"

echo "=========================================================="
echo "Placebo sweep for:"
echo "$BASE_CMD"
echo "=========================================================="

echo "Running baseline (no reg)..."
eval "$BASE_CMD --paired_seeds --result_dir $RESULT_DIR"

for transform in none shuffle homophily
do
    for lambda_val in 0.05 0.1 0.2 0.4
    do
        echo ""
        echo "Running mlp_reg | transform=$transform | lambda=$lambda_val ..."
        eval "$BASE_CMD --paired_seeds --use_reg --mlp_reg --mlp_epochs 500 \
              --lambda_val $lambda_val --penalty_transform $transform \
              --result_dir $RESULT_DIR"
    done
done

echo ""
echo "Placebo sweep completed!"
