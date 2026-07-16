#!/bin/bash -l
#SBATCH --job-name=placebo
#SBATCH --output=results/placebo_%A_%a.log
#SBATCH --error=results/placebo_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-47
#SBATCH --requeue

# Placebo experiment: does the co-occurrence penalty help because of the
# INFORMATION it carries, or because any penalty of this form regularizes?
# Conditions per (dataset, model): baseline, true penalty, shuffled penalty
# (placebo), homophily-only penalty (structure ablation), oracle penalty
# (ceiling), each over the lambda grid, 10 paired runs. See PLACEBO_README.md.
#
# Each pair is split into 6 condition groups so the heavy 2500-epoch tasks
# (amazon-ratings, roman-empire: up to 150k training epochs per group) stay
# safely inside the 48h limit:  task_id = pair_idx * 6 + group
#   group 0: baseline + mlp/none x lambda grid   (6 configs)
#   group 1: mlp/shuffle x lambda grid           (5 configs)
#   group 2: mlp/homophily x lambda grid         (5 configs)
#   group 3: oracle x lambda grid                (5 configs)
#   group 4: mlp/none x lambda, mlp_epochs=10    (5 configs, E4.3 quality curve)
#   group 5: mlp/none x lambda, mlp_epochs=50    (5 configs, E4.3 quality curve)
# Pairing across groups holds because --paired_seeds makes run k
# deterministic from the seed, independent of which task runs it.
#
# Targets: the (dataset, model) pairs where FINDINGS.md Table 2 reports the
# largest mlp_gnn wins, using the tunedGNN-optimized hyperparameters from
# run_gnn.sh verbatim. roman-empire/GCN is EXCLUDED (degenerate baseline,
# 28.54 vs ~91 published — must be fixed before it is interpretable).
#
# Must be submitted from medium_graph/.

set -e

DEVICE=0
RUNS=10

echo "=========================================================="
echo "Placebo sweep | Task $SLURM_ARRAY_TASK_ID"
echo "Job: $SLURM_ARRAY_JOB_ID | Date: $(date) | Host: $(hostname)"
echo "=========================================================="

mkdir -p results

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /project/ikoutis/conda_env/co-occurrence

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

PAIR_IDX=$((SLURM_ARRAY_TASK_ID / 6))
GROUP=$((SLURM_ARRAY_TASK_ID % 6))

# Hyperparameters copied verbatim from run_gnn.sh (tunedGNN-optimized),
# with --runs $RUNS and --device $DEVICE.
case $PAIR_IDX in
  0)
    CMD="python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE --runs $RUNS"
    ;;
  1)
    CMD="python main.py --gnn gat --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.2 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE --runs $RUNS --res"
    ;;
  2)
    CMD="python main.py --gnn gat --dataset citeseer --lr 0.001 --local_layers 3 --hidden_channels 256 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $DEVICE --runs $RUNS --res"
    ;;
  3)
    CMD="python main.py --gnn gat --dataset squirrel --lr 0.005 --local_layers 7 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.5 --device $DEVICE --runs $RUNS --bn --res"
    ;;
  4)
    CMD="python main.py --gnn gcn --dataset squirrel --lr 0.01 --local_layers 4 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --device $DEVICE --runs $RUNS --bn --res"
    ;;
  5)
    CMD="python main.py --gnn sage --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 9 --weight_decay 0.0 --dropout 0.5 --device $DEVICE --runs $RUNS --bn --res"
    ;;
  6)
    CMD="python main.py --gnn gcn --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 4 --weight_decay 0.0 --dropout 0.5 --device $DEVICE --runs $RUNS --bn --res"
    ;;
  7)
    CMD="python main.py --gnn gat --dataset roman-empire --pre_linear --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 10 --weight_decay 0.0 --dropout 0.3 --device $DEVICE --runs $RUNS --bn --res"
    ;;
  *)
    echo "Unknown pair idx $PAIR_IDX (task $SLURM_ARRAY_TASK_ID)"; exit 1
    ;;
esac

case $GROUP in
  0) ./sweep_conditions.sh "$CMD" results/placebo "none"      0 1 ;;
  1) ./sweep_conditions.sh "$CMD" results/placebo "shuffle"   0 0 ;;
  2) ./sweep_conditions.sh "$CMD" results/placebo "homophily" 0 0 ;;
  3) ./sweep_conditions.sh "$CMD" results/placebo ""          1 0 ;;
  4) ./sweep_conditions.sh "$CMD" results/placebo "none"      0 0 10 ;;
  5) ./sweep_conditions.sh "$CMD" results/placebo "none"      0 0 50 ;;
esac

echo "Task $SLURM_ARRAY_TASK_ID done."
