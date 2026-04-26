#!/bin/bash -l
#SBATCH --job-name=gnn_mlp_gnn
#SBATCH --output=results/gnn_mlp_gnn_%A_%a.log
#SBATCH --error=results/gnn_mlp_gnn_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=72:00:00
#SBATCH --array=0-13
#SBATCH --requeue

set -e

DATASETS=(amazon-ratings minesweeper roman-empire questions squirrel chameleon amazon-computer amazon-photo coauthor-cs coauthor-physics wikics cora citeseer pubmed)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
DEVICE=0
RUNS=5
RD=results/mlp_gnn

echo "=========================================================="
echo "MLP-GNN Sweep — Dataset: $DATASET"
echo "Array task: $SLURM_ARRAY_TASK_ID / Job: $SLURM_ARRAY_JOB_ID"
echo "Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p results

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

case $DATASET in
  amazon-ratings)
    python main.py --model MLP --dataset amazon-ratings --hidden_channels 512 --local_layers 4 --dropout 0.5 --lr 0.001 --weight_decay 5e-4 --epochs 1000 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset amazon-ratings --hidden_channels 512 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 4  --weight_decay 5e-4 --dropout 0.5 --device $DEVICE --bn --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset amazon-ratings --hidden_channels 512 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 9  --weight_decay 5e-4 --dropout 0.5 --device $DEVICE --bn --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset amazon-ratings --hidden_channels 512 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 4  --weight_decay 5e-4 --dropout 0.5 --device $DEVICE --bn --res"
    ;;
  minesweeper)
    python main.py --model MLP --dataset minesweeper --hidden_channels 64 --local_layers 4 --dropout 0.2 --lr 0.01 --weight_decay 5e-4 --epochs 1000 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset minesweeper --hidden_channels 64  --epochs 1000 --lr 0.01  --runs $RUNS --local_layers 12 --weight_decay 5e-4 --dropout 0.2 --device $DEVICE --bn --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset minesweeper --hidden_channels 64  --epochs 1000 --lr 0.01  --runs $RUNS --local_layers 15 --weight_decay 5e-4 --dropout 0.2 --device $DEVICE --bn --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset minesweeper --hidden_channels 64  --epochs 1000 --lr 0.01  --runs $RUNS --local_layers 15 --weight_decay 5e-4 --dropout 0.2 --device $DEVICE --bn --res"
    ;;
  roman-empire)
    python main.py --model MLP --dataset roman-empire --hidden_channels 512 --local_layers 4 --dropout 0.5 --lr 0.001 --weight_decay 5e-4 --epochs 1000 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset roman-empire --hidden_channels 512 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 9  --weight_decay 5e-4 --dropout 0.5 --device $DEVICE --bn --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset roman-empire --hidden_channels 256 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 9  --weight_decay 5e-4 --dropout 0.3 --device $DEVICE --bn"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset roman-empire --hidden_channels 512 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 10 --weight_decay 5e-4 --dropout 0.3 --device $DEVICE --bn --res"
    ;;
  questions)
    python main.py --model MLP --dataset questions --hidden_channels 512 --local_layers 4 --dropout 0.3 --lr 3e-5 --weight_decay 5e-4 --epochs 1000 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset questions --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs $RUNS --local_layers 10 --weight_decay 5e-4 --dropout 0.3 --device $DEVICE --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset questions --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs $RUNS --local_layers 6  --weight_decay 5e-4 --dropout 0.2 --device $DEVICE --ln"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset questions --hidden_channels 512 --epochs 1000 --lr 3e-5 --runs $RUNS --local_layers 3  --weight_decay 5e-4 --dropout 0.2 --device $DEVICE --ln --res"
    ;;
  squirrel)
    python main.py --model MLP --dataset squirrel --hidden_channels 256 --local_layers 4 --dropout 0.7 --lr 0.01 --weight_decay 5e-4 --epochs 1000 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset squirrel --hidden_channels 256 --epochs 1000 --lr 0.01  --runs $RUNS --local_layers 4  --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --ln --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset squirrel --hidden_channels 256 --epochs 1000 --lr 0.01  --runs $RUNS --local_layers 3  --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --ln --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset squirrel --hidden_channels 512 --epochs 1000 --lr 0.005 --runs $RUNS --local_layers 7  --weight_decay 5e-4 --dropout 0.5 --device $DEVICE --ln --res"
    ;;
  chameleon)
    python main.py --model MLP --dataset chameleon --hidden_channels 512 --local_layers 4 --dropout 0.2 --lr 0.005 --weight_decay 5e-4 --epochs 1000 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset chameleon --hidden_channels 512 --epochs 1000 --lr 0.005 --runs $RUNS --local_layers 5  --weight_decay 5e-4 --dropout 0.2 --device $DEVICE"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset chameleon --hidden_channels 256 --epochs 1000 --lr 0.01  --runs $RUNS --local_layers 4  --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --ln --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset chameleon --hidden_channels 256 --epochs 1000 --lr 0.01  --runs $RUNS --local_layers 2  --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --ln --res"
    ;;
  amazon-computer)
    python main.py --model MLP --dataset amazon-computer --hidden_channels 512 --local_layers 3 --dropout 0.5 --lr 0.001 --weight_decay 5e-5 --epochs 1000 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset amazon-computer --hidden_channels 512 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device $DEVICE --ln"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset amazon-computer --hidden_channels 64  --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 4 --weight_decay 5e-5 --dropout 0.3 --device $DEVICE --ln"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset amazon-computer --hidden_channels 64  --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 2 --weight_decay 5e-5 --dropout 0.5 --device $DEVICE --ln"
    ;;
  amazon-photo)
    python main.py --model MLP --dataset amazon-photo --hidden_channels 256 --local_layers 3 --dropout 0.5 --lr 0.001 --weight_decay 5e-5 --epochs 1000 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset amazon-photo --hidden_channels 256 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device $DEVICE --ln --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset amazon-photo --hidden_channels 64  --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 6 --weight_decay 5e-5 --dropout 0.2 --device $DEVICE --ln --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset amazon-photo --hidden_channels 64  --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device $DEVICE --ln --res"
    ;;
  coauthor-cs)
    python main.py --model MLP --dataset coauthor-cs --hidden_channels 512 --local_layers 3 --dropout 0.3 --lr 0.001 --weight_decay 5e-4 --epochs 1500 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset coauthor-cs --hidden_channels 512 --epochs 1500 --lr 0.001 --runs $RUNS --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device $DEVICE --ln --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset coauthor-cs --hidden_channels 512 --epochs 1500 --lr 0.001 --runs $RUNS --local_layers 2 --weight_decay 5e-4 --dropout 0.5 --device $DEVICE --ln --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset coauthor-cs --hidden_channels 256 --epochs 1500 --lr 0.001 --runs $RUNS --local_layers 1 --weight_decay 5e-4 --dropout 0.3 --device $DEVICE --ln --res"
    ;;
  coauthor-physics)
    python main.py --model MLP --dataset coauthor-physics --hidden_channels 64 --local_layers 3 --dropout 0.3 --lr 0.001 --weight_decay 5e-4 --epochs 1500 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset coauthor-physics --hidden_channels 64  --epochs 1500 --lr 0.001 --runs $RUNS --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device $DEVICE --ln --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset coauthor-physics --hidden_channels 64  --epochs 1500 --lr 0.001 --runs $RUNS --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --bn --res"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset coauthor-physics --hidden_channels 256 --epochs 1500 --lr 0.001 --runs $RUNS --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --bn --res"
    ;;
  wikics)
    python main.py --model MLP --dataset wikics --hidden_channels 256 --local_layers 3 --dropout 0.5 --lr 0.001 --weight_decay 0.0 --epochs 1000 --runs $RUNS --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset wikics --hidden_channels 256 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device $DEVICE --ln"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset wikics --hidden_channels 256 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 2 --weight_decay 0.0 --dropout 0.7 --device $DEVICE --ln"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset wikics --hidden_channels 512 --epochs 1000 --lr 0.001 --runs $RUNS --local_layers 2 --weight_decay 0.0 --dropout 0.7 --device $DEVICE --ln --res"
    ;;
  cora)
    python main.py --model MLP --dataset cora --hidden_channels 512 --local_layers 3 --dropout 0.7 --lr 0.001 --weight_decay 5e-4 --epochs 1000 --runs $RUNS --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset cora --hidden_channels 512 --lr 0.001 --runs $RUNS --local_layers 3 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset cora --hidden_channels 256 --lr 0.001 --runs $RUNS --local_layers 3 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset cora --hidden_channels 512 --lr 0.001 --runs $RUNS --local_layers 3 --weight_decay 5e-4 --dropout 0.2 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE --res"
    ;;
  citeseer)
    python main.py --model MLP --dataset citeseer --hidden_channels 512 --local_layers 3 --dropout 0.5 --lr 0.001 --weight_decay 0.01 --epochs 1000 --runs $RUNS --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset citeseer --hidden_channels 512 --lr 0.001 --runs $RUNS --local_layers 2 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset citeseer --hidden_channels 512 --lr 0.001 --runs $RUNS --local_layers 3 --weight_decay 0.01 --dropout 0.2 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset citeseer --hidden_channels 256 --lr 0.001 --runs $RUNS --local_layers 3 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE --res"
    ;;
  pubmed)
    python main.py --model MLP --dataset pubmed --hidden_channels 256 --local_layers 3 --dropout 0.7 --lr 0.005 --weight_decay 5e-4 --epochs 1000 --runs $RUNS --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE --result_dir $RD
    ./sweep_mlp_gnn.sh "python main.py --gnn gcn  --dataset pubmed --hidden_channels 256 --lr 0.005 --runs $RUNS --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_mlp_gnn.sh "python main.py --gnn sage --dataset pubmed --hidden_channels 512 --lr 0.005 --runs $RUNS --local_layers 4 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_mlp_gnn.sh "python main.py --gnn gat  --dataset pubmed --hidden_channels 512 --lr 0.01  --runs $RUNS --local_layers 2 --weight_decay 5e-4 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ;;
esac

echo "=========================================================="
echo "Sweep for $DATASET completed at: $(date)"
echo "=========================================================="
