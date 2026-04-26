#!/bin/bash -l
#SBATCH --job-name=gnn_mlp_homo
#SBATCH --output=results/gnn_mlp_homo_%A_%a.log
#SBATCH --error=results/gnn_mlp_homo_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --requeue

set -e

DATASETS=(amazon-computer amazon-photo coauthor-cs coauthor-physics wikics cora citeseer pubmed)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
DEVICE=0

echo "=========================================================="
echo "Homophilic MLP-Reg GNN Sweep — Dataset: $DATASET"
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
  amazon-computer)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset amazon-computer --hidden_channels 512 --epochs 1000 --lr 0.001 --runs 3 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device $DEVICE --ln"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset amazon-computer --hidden_channels 64  --epochs 1000 --lr 0.001 --runs 3 --local_layers 4 --weight_decay 5e-5 --dropout 0.3 --device $DEVICE --ln"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset amazon-computer --hidden_channels 64  --epochs 1000 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-5 --dropout 0.5 --device $DEVICE --ln"
    ;;
  amazon-photo)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset amazon-photo --hidden_channels 256 --epochs 1000 --lr 0.001 --runs 3 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device $DEVICE --ln --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset amazon-photo --hidden_channels 64  --epochs 1000 --lr 0.001 --runs 3 --local_layers 6 --weight_decay 5e-5 --dropout 0.2 --device $DEVICE --ln --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset amazon-photo --hidden_channels 64  --epochs 1000 --lr 0.001 --runs 3 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device $DEVICE --ln --res"
    ;;
  coauthor-cs)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset coauthor-cs --hidden_channels 512 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device $DEVICE --ln --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset coauthor-cs --hidden_channels 512 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.5 --device $DEVICE --ln --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset coauthor-cs --hidden_channels 256 --epochs 1500 --lr 0.001 --runs 3 --local_layers 1 --weight_decay 5e-4 --dropout 0.3 --device $DEVICE --ln --res"
    ;;
  coauthor-physics)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset coauthor-physics --hidden_channels 64  --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device $DEVICE --ln --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset coauthor-physics --hidden_channels 64  --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --bn --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset coauthor-physics --hidden_channels 256 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --bn --res"
    ;;
  wikics)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset wikics --hidden_channels 256 --epochs 1000 --lr 0.001 --runs 3 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device $DEVICE --ln"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset wikics --hidden_channels 256 --epochs 1000 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 0.0 --dropout 0.7 --device $DEVICE --ln"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset wikics --hidden_channels 512 --epochs 1000 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 0.0 --dropout 0.7 --device $DEVICE --ln --res"
    ;;
  cora)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset cora --hidden_channels 512 --lr 0.001 --runs 5 --local_layers 3 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset cora --hidden_channels 256 --lr 0.001 --runs 5 --local_layers 3 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset cora --hidden_channels 512 --lr 0.001 --runs 5 --local_layers 3 --weight_decay 5e-4 --dropout 0.2 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE --res"
    ;;
  citeseer)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset citeseer --hidden_channels 512 --lr 0.001 --runs 5 --local_layers 2 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset citeseer --hidden_channels 512 --lr 0.001 --runs 5 --local_layers 3 --weight_decay 0.01 --dropout 0.2 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset citeseer --hidden_channels 256 --lr 0.001 --runs 5 --local_layers 3 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE --res"
    ;;
  pubmed)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset pubmed --hidden_channels 256 --lr 0.005 --runs 5 --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset pubmed --hidden_channels 512 --lr 0.005 --runs 5 --local_layers 4 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset pubmed --hidden_channels 512 --lr 0.01  --runs 5 --local_layers 2 --weight_decay 5e-4 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE"
    ;;
esac

echo "=========================================================="
echo "Sweep for $DATASET completed at: $(date)"
echo "=========================================================="
