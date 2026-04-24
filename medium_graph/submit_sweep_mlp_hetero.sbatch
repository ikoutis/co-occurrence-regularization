#!/bin/bash -l
#SBATCH --job-name=gnn_mlp_hetero
#SBATCH --output=results/gnn_mlp_hetero_%A_%a.log
#SBATCH --error=results/gnn_mlp_hetero_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --requeue

set -e

DATASETS=(amazon-ratings minesweeper roman-empire questions squirrel chameleon)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
DEVICE=0

echo "=========================================================="
echo "Heterophilic MLP-Reg GNN Sweep — Dataset: $DATASET"
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
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 4  --weight_decay 0.0   --dropout 0.5 --device $DEVICE --bn --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 9  --weight_decay 0.0   --dropout 0.5 --device $DEVICE --bn --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 4  --weight_decay 0.0   --dropout 0.5 --device $DEVICE --bn --res"
    ;;
  minesweeper)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 3 --local_layers 12 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device $DEVICE --bn --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 3 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device $DEVICE --bn --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 3 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device $DEVICE --bn --res"
    ;;
  roman-empire)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset roman-empire --pre_linear --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 9  --weight_decay 0.0 --dropout 0.5 --device $DEVICE --bn --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset roman-empire --pre_linear --hidden_channels 256 --epochs 2500 --lr 0.001 --runs 3 --local_layers 9  --weight_decay 0.0 --dropout 0.3 --device $DEVICE --bn"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset roman-empire --pre_linear --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 10 --weight_decay 0.0 --dropout 0.3 --device $DEVICE --bn --res"
    ;;
  questions)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 3 --local_layers 10 --weight_decay 0.0 --dropout 0.3 --metric rocauc --device $DEVICE --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 3 --local_layers 6  --weight_decay 0.0 --dropout 0.2 --metric rocauc --device $DEVICE --ln"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 3 --local_layers 3  --weight_decay 0.0 --dropout 0.2 --metric rocauc --device $DEVICE --ln --res"
    ;;
  squirrel)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset squirrel --lr 0.01  --local_layers 4 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --runs 10 --bn --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset squirrel --lr 0.01  --local_layers 3 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --runs 10 --bn --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset squirrel --lr 0.005 --local_layers 7 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.5 --device $DEVICE --runs 10 --bn --res"
    ;;
  chameleon)
    ./sweep_lambda_mlp.sh "python main.py --gnn gcn  --dataset chameleon --lr 0.005 --local_layers 5 --hidden_channels 512 --weight_decay 0.001 --dropout 0.2 --device $DEVICE --runs 10 --epochs 200"
    ./sweep_lambda_mlp.sh "python main.py --gnn sage --dataset chameleon --lr 0.01  --local_layers 4 --hidden_channels 256 --weight_decay 0.001 --dropout 0.7 --device $DEVICE --runs 10 --epochs 200 --bn --res"
    ./sweep_lambda_mlp.sh "python main.py --gnn gat  --dataset chameleon --lr 0.01  --local_layers 2 --hidden_channels 256 --weight_decay 0.001 --dropout 0.7 --device $DEVICE --runs 10 --epochs 200 --bn --res"
    ;;
esac

echo "=========================================================="
echo "Sweep for $DATASET completed at: $(date)"
echo "=========================================================="
