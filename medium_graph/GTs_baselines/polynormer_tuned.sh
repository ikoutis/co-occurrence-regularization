#!/bin/bash

# Polynormer PUBLISHED per-dataset configs, taken verbatim from the official
# repo (cornell-zhang/Polynormer, run.sh) and mapped to this codebase's flags.
# Requires the two-phase schedule (--local_epochs/--global_epochs) added to
# main.py — without it, Polynormer's global attention is never trained.
# --save_model is required: the official schedule warm-starts the global
# phase from the best-validation local checkpoint.
#
# Expected results (official repo README, single run):
#   roman-empire 92.48 | amazon-ratings 55.04 | minesweeper 97.19 (ROC)
#   questions 78.35 (ROC) | amazon-computer 94.07 | amazon-photo 96.67
#   coauthor-cs 95.28 | coauthor-physics 97.14 | wikics 81.20
#
# Usage: ./polynormer_tuned.sh <device> [runs] [extra flags...]

DEVICE=${1:-0}
RUNS=${2:-10}
EXTRA="${@:3}"

## heterophilic datasets
python main.py --model polynormer --dataset roman-empire --hidden_channels 64 --local_epochs 100 --global_epochs 2500 --lr 0.001 --runs $RUNS --local_layers 10 --global_layers 2 --weight_decay 0.0 --dropout 0.3 --global_dropout 0.5 --in_dropout 0.15 --num_heads 8 --beta 0.5 --save_model --device $DEVICE $EXTRA
python main.py --model polynormer --dataset amazon-ratings --hidden_channels 256 --local_epochs 200 --global_epochs 2500 --lr 0.001 --runs $RUNS --local_layers 10 --global_layers 1 --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 2 --save_model --device $DEVICE $EXTRA
python main.py --model polynormer --dataset minesweeper --hidden_channels 64 --local_epochs 100 --global_epochs 2000 --lr 0.001 --runs $RUNS --local_layers 10 --global_layers 3 --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 8 --metric rocauc --save_model --device $DEVICE $EXTRA
python main.py --model polynormer --dataset questions --hidden_channels 64 --local_epochs 200 --global_epochs 1500 --lr 3e-5 --runs $RUNS --local_layers 5 --global_layers 3 --weight_decay 0.0 --dropout 0.2 --global_dropout 0.5 --in_dropout 0.15 --num_heads 8 --beta 0.4 --pre_ln --metric rocauc --save_model --device $DEVICE $EXTRA

## homophilic datasets
python main.py --model polynormer --dataset amazon-computer --hidden_channels 64 --local_epochs 200 --global_epochs 1000 --lr 0.001 --runs $RUNS --local_layers 5 --global_layers 1 --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --save_model --device $DEVICE $EXTRA
python main.py --model polynormer --dataset amazon-photo --hidden_channels 64 --local_epochs 200 --global_epochs 1000 --lr 0.001 --runs $RUNS --local_layers 7 --global_layers 2 --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --save_model --device $DEVICE $EXTRA
python main.py --model polynormer --dataset coauthor-cs --hidden_channels 64 --local_epochs 100 --global_epochs 1500 --lr 0.001 --runs $RUNS --local_layers 5 --global_layers 2 --weight_decay 5e-4 --dropout 0.3 --in_dropout 0.1 --num_heads 8 --save_model --device $DEVICE $EXTRA
python main.py --model polynormer --dataset coauthor-physics --hidden_channels 32 --local_epochs 100 --global_epochs 1500 --lr 0.001 --runs $RUNS --local_layers 5 --global_layers 4 --weight_decay 5e-4 --dropout 0.5 --in_dropout 0.1 --num_heads 8 --save_model --device $DEVICE $EXTRA
python main.py --model polynormer --dataset wikics --hidden_channels 512 --local_epochs 100 --global_epochs 1000 --lr 0.001 --runs $RUNS --local_layers 7 --global_layers 2 --weight_decay 0.0 --dropout 0.5 --in_dropout 0.5 --num_heads 1 --save_model --device $DEVICE $EXTRA
