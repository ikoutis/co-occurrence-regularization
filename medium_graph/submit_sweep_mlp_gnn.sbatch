#!/bin/bash -l
#SBATCH --job-name=mlp_gnn
#SBATCH --output=results/mlp_gnn_%A_%a.log
#SBATCH --error=results/mlp_gnn_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-55
#SBATCH --requeue

set -e

# 14 datasets x 4 methods (MLP, gcn, sage, gat) = 56 tasks
# task_id  -> (dataset, method)  via:  d_idx = task % 14, m_idx = task / 14
DATASETS=(amazon-ratings minesweeper roman-empire questions squirrel chameleon \
          amazon-computer amazon-photo coauthor-cs coauthor-physics wikics \
          cora citeseer pubmed)
METHODS=(MLP gcn sage gat)

D_IDX=$(( SLURM_ARRAY_TASK_ID % 14 ))
M_IDX=$(( SLURM_ARRAY_TASK_ID / 14 ))
DATASET=${DATASETS[$D_IDX]}
METHOD=${METHODS[$M_IDX]}

DEVICE=0
RUNS=5
RD=results/mlp_gnn

echo "=========================================================="
echo "MLP-GNN Task $SLURM_ARRAY_TASK_ID -> Dataset: $DATASET | Method: $METHOD"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p results

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

run_mlp() {
    python main.py --model MLP --dataset "$1" --hidden_channels "$2" --local_layers "$3" \
        --dropout "$4" --lr "$5" --weight_decay "$6" --epochs "$7" --runs $RUNS \
        --device $DEVICE --result_dir $RD ${@:8}
}

run_gnn() {
    ./sweep_mlp_gnn.sh "python main.py --gnn $1 --dataset $2 --hidden_channels $3 \
        --epochs $4 --lr $5 --runs $RUNS --local_layers $6 --weight_decay $7 \
        --dropout $8 --device $DEVICE ${@:9}"
}

case "$DATASET--$METHOD" in
  # ---------------- amazon-ratings ----------------
  amazon-ratings--MLP)  run_mlp amazon-ratings 512 4 0.5 0.001 5e-4 1000 ;;
  amazon-ratings--gcn)  run_gnn gcn  amazon-ratings 512 1000 0.001 4 5e-4 0.5 --bn --res ;;
  amazon-ratings--sage) run_gnn sage amazon-ratings 512 1000 0.001 9 5e-4 0.5 --bn --res ;;
  amazon-ratings--gat)  run_gnn gat  amazon-ratings 512 1000 0.001 4 5e-4 0.5 --bn --res ;;

  # ---------------- minesweeper (binary -> rocauc) ----------------
  minesweeper--MLP)  run_mlp minesweeper 64 4 0.2 0.01 5e-4 1000 --metric rocauc ;;
  minesweeper--gcn)  run_gnn gcn  minesweeper 64 1000 0.01 12 5e-4 0.2 --bn --res --metric rocauc ;;
  minesweeper--sage) run_gnn sage minesweeper 64 1000 0.01 15 5e-4 0.2 --bn --res --metric rocauc ;;
  minesweeper--gat)  run_gnn gat  minesweeper 64 1000 0.01 15 5e-4 0.2 --bn --res --metric rocauc ;;

  # ---------------- roman-empire ----------------
  roman-empire--MLP)  run_mlp roman-empire 512 4 0.5 0.001 5e-4 1000 ;;
  roman-empire--gcn)  run_gnn gcn  roman-empire 512 1000 0.001 9  5e-4 0.5 --bn --res ;;
  roman-empire--sage) run_gnn sage roman-empire 256 1000 0.001 9  5e-4 0.3 --bn ;;
  roman-empire--gat)  run_gnn gat  roman-empire 512 1000 0.001 10 5e-4 0.3 --bn --res ;;

  # ---------------- questions (binary -> rocauc) ----------------
  questions--MLP)  run_mlp questions 512 4 0.3 3e-5 5e-4 1000 --metric rocauc ;;
  questions--gcn)  run_gnn gcn  questions 512 1000 3e-5 10 5e-4 0.3 --res --metric rocauc ;;
  questions--sage) run_gnn sage questions 512 1000 3e-5 6  5e-4 0.2 --ln --metric rocauc ;;
  questions--gat)  run_gnn gat  questions 512 1000 3e-5 3  5e-4 0.2 --ln --res --metric rocauc ;;

  # ---------------- squirrel ----------------
  squirrel--MLP)  run_mlp squirrel 256 4 0.7 0.01 5e-4 1000 ;;
  squirrel--gcn)  run_gnn gcn  squirrel 256 1000 0.01  4 5e-4 0.7 --ln --res ;;
  squirrel--sage) run_gnn sage squirrel 256 1000 0.01  3 5e-4 0.7 --ln --res ;;
  squirrel--gat)  run_gnn gat  squirrel 512 1000 0.005 7 5e-4 0.5 --ln --res ;;

  # ---------------- chameleon ----------------
  chameleon--MLP)  run_mlp chameleon 512 4 0.2 0.005 5e-4 1000 ;;
  chameleon--gcn)  run_gnn gcn  chameleon 512 1000 0.005 5 5e-4 0.2 ;;
  chameleon--sage) run_gnn sage chameleon 256 1000 0.01  4 5e-4 0.7 --ln --res ;;
  chameleon--gat)  run_gnn gat  chameleon 256 1000 0.01  2 5e-4 0.7 --ln --res ;;

  # ---------------- amazon-computer ----------------
  amazon-computer--MLP)  run_mlp amazon-computer 512 3 0.5 0.001 5e-5 1000 ;;
  amazon-computer--gcn)  run_gnn gcn  amazon-computer 512 1000 0.001 3 5e-5 0.5 --ln ;;
  amazon-computer--sage) run_gnn sage amazon-computer 64  1000 0.001 4 5e-5 0.3 --ln ;;
  amazon-computer--gat)  run_gnn gat  amazon-computer 64  1000 0.001 2 5e-5 0.5 --ln ;;

  # ---------------- amazon-photo ----------------
  amazon-photo--MLP)  run_mlp amazon-photo 256 3 0.5 0.001 5e-5 1000 ;;
  amazon-photo--gcn)  run_gnn gcn  amazon-photo 256 1000 0.001 6 5e-5 0.5 --ln --res ;;
  amazon-photo--sage) run_gnn sage amazon-photo 64  1000 0.001 6 5e-5 0.2 --ln --res ;;
  amazon-photo--gat)  run_gnn gat  amazon-photo 64  1000 0.001 3 5e-5 0.5 --ln --res ;;

  # ---------------- coauthor-cs ----------------
  coauthor-cs--MLP)  run_mlp coauthor-cs 512 3 0.3 0.001 5e-4 1500 ;;
  coauthor-cs--gcn)  run_gnn gcn  coauthor-cs 512 1500 0.001 2 5e-4 0.3 --ln --res ;;
  coauthor-cs--sage) run_gnn sage coauthor-cs 512 1500 0.001 2 5e-4 0.5 --ln --res ;;
  coauthor-cs--gat)  run_gnn gat  coauthor-cs 256 1500 0.001 1 5e-4 0.3 --ln --res ;;

  # ---------------- coauthor-physics ----------------
  coauthor-physics--MLP)  run_mlp coauthor-physics 64 3 0.3 0.001 5e-4 1500 ;;
  coauthor-physics--gcn)  run_gnn gcn  coauthor-physics 64  1500 0.001 2 5e-4 0.3 --ln --res ;;
  coauthor-physics--sage) run_gnn sage coauthor-physics 64  1500 0.001 2 5e-4 0.7 --bn --res ;;
  coauthor-physics--gat)  run_gnn gat  coauthor-physics 256 1500 0.001 2 5e-4 0.7 --bn --res ;;

  # ---------------- wikics ----------------
  wikics--MLP)  run_mlp wikics 256 3 0.5 0.001 0.0 1000 ;;
  wikics--gcn)  run_gnn gcn  wikics 256 1000 0.001 3 0.0 0.5 --ln ;;
  wikics--sage) run_gnn sage wikics 256 1000 0.001 2 0.0 0.7 --ln ;;
  wikics--gat)  run_gnn gat  wikics 512 1000 0.001 2 0.0 0.7 --ln --res ;;

  # ---------------- cora ----------------
  cora--MLP)  run_mlp cora 512 3 0.7 0.001 5e-4 1000 \
                --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  cora--gcn)  run_gnn gcn  cora 512 1000 0.001 3 5e-4 0.7 \
                --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  cora--sage) run_gnn sage cora 256 1000 0.001 3 5e-4 0.7 \
                --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  cora--gat)  run_gnn gat  cora 512 1000 0.001 3 5e-4 0.2 \
                --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --res ;;

  # ---------------- citeseer ----------------
  citeseer--MLP)  run_mlp citeseer 512 3 0.5 0.001 0.01 1000 \
                    --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  citeseer--gcn)  run_gnn gcn  citeseer 512 1000 0.001 2 0.01 0.5 \
                    --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  citeseer--sage) run_gnn sage citeseer 512 1000 0.001 3 0.01 0.2 \
                    --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  citeseer--gat)  run_gnn gat  citeseer 256 1000 0.001 3 0.01 0.5 \
                    --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --res ;;

  # ---------------- pubmed ----------------
  pubmed--MLP)  run_mlp pubmed 256 3 0.7 0.005 5e-4 1000 \
                  --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  pubmed--gcn)  run_gnn gcn  pubmed 256 1000 0.005 2 5e-4 0.7 \
                  --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  pubmed--sage) run_gnn sage pubmed 512 1000 0.005 4 5e-4 0.7 \
                  --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;
  pubmed--gat)  run_gnn gat  pubmed 512 1000 0.01  2 5e-4 0.5 \
                  --rand_split_class --valid_num 500 --test_num 1000 --seed 123 ;;

  *)
    echo "ERROR: unknown task $DATASET--$METHOD"
    exit 1
    ;;
esac

echo "=========================================================="
echo "Task $SLURM_ARRAY_TASK_ID ($DATASET--$METHOD) completed at: $(date)"
echo "=========================================================="
