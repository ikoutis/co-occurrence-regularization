#!/bin/bash

# SGFormer PUBLISHED per-dataset configs, taken verbatim from the official
# repo (qitianwu/SGFormer, medium/run.sh, --method ours) and mapped to this
# codebase's flags:
#   official --num_layers        -> --layers      (GCN backbone depth)
#   official --ours_layers       -> --tr_layers   (transformer depth)
#   official --ours_dropout      -> --tr_dropout
#   official --ours_weight_decay -> --tr_weight_decay (two-group optimizer)
#   official --no_feat_norm      -> loader default here (no flag needed)
#   official --ours_use_residual -> --use_residual (trans-branch residual).
#     NOTE the official '--use_residual' flag is a NO-OP for method=ours —
#     the model receives use_residual=args.ours_use_residual — so the
#     citation configs run the transformer branch WITHOUT residual and only
#     chameleon (which passes --ours_use_residual) runs it WITH residual.
#
# Published results (SGFormer paper, 20-per-class splits):
#   cora 84.5 | citeseer 72.6 | pubmed 80.3
# chameleon: both the official repo and this repo load the FILTERED dataset
# (dataset.py reads chameleon_filtered.npz), so the published number (~44.9)
# is directly comparable.
# squirrel is NOT here: the official config uses --method difformer (a
# different attention), not reproducible with this SGFormer implementation —
# it goes through the search instead (submit_gt_search.sbatch).
#
# Usage: ./sgformer_tuned.sh <device> [runs] [extra flags...]

DEVICE=${1:-0}
RUNS=${2:-10}
EXTRA="${@:3}"

python main.py --model sgformer --dataset cora --lr 0.01 --layers 4 --tr_layers 1 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 --tr_dropout 0.2 --tr_weight_decay 0.001 --use_graph --graph_weight 0.8 --alpha 0.5 --num_heads 1 --epochs 500 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --runs $RUNS --device $DEVICE $EXTRA

python main.py --model sgformer --dataset citeseer --lr 0.005 --layers 4 --tr_layers 1 --hidden_channels 64 --weight_decay 0.01 --dropout 0.5 --tr_dropout 0.3 --tr_weight_decay 0.01 --use_graph --graph_weight 0.7 --alpha 0.5 --num_heads 1 --epochs 500 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --runs $RUNS --device $DEVICE $EXTRA

python main.py --model sgformer --dataset pubmed --lr 0.005 --layers 4 --tr_layers 1 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 --tr_dropout 0.3 --tr_weight_decay 0.01 --use_graph --graph_weight 0.8 --alpha 0.5 --num_heads 1 --epochs 500 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --runs $RUNS --device $DEVICE $EXTRA

python main.py --model sgformer --dataset chameleon --lr 0.001 --layers 2 --tr_layers 1 --hidden_channels 64 --weight_decay 0.001 --dropout 0.6 --use_graph --use_residual --alpha 0.5 --num_heads 1 --epochs 200 --runs $RUNS --device $DEVICE $EXTRA
