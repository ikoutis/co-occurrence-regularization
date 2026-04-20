# GNN Regularization: Testing Plan & Hand-off

This document serves as a follow-up and execution guide for running the newly implemented dynamic structure-aware regularization on a new machine.

## What has been accomplished
1. **Dynamic Regularization Implemented:** The core logic (`estimate_cooccurrence_matrix` and `edge_loss`) was successfully integrated into the main training loop for both the classic MPNNs (`medium_graph/main.py`) and the Graph Transformers (`medium_graph/GTs_baselines/main*.py`).
2. **Logger Updated:** Both logging modules (`logger.py`) have been updated to explicitly record whether regularization was used and which lambda value was selected (e.g., `REG: 0.5` or `REG: False`). This makes it easy to parse the results from the CSVs.
3. **Utility Scripts Created:** 
   - Created `sweep_lambda.sh` to allow decoupled benchmarking (sweeping only the `lambda` values over the best pre-determined model configurations).
   - Created `run_gnn_sweep.sh` which wraps all 40+ optimal MPNN configurations from the original paper in the new lambda sweep script.

## Immediate Plan: Testing Classic GNNs (Phase 1)

Since you are transitioning to a machine with 4GB VRAM, we will focus solely on the classic GNNs (GCN, GraphSAGE, GAT). These models (mostly) fit within your memory budget and their optimal configurations are already known and locked in.

### Instructions to run the sweep

1. **Navigate to the medium graph directory:**
   ```bash
   cd medium_graph
   ```

2. **Execute the Decoupled GNN Sweep:**
   The `run_gnn_sweep.sh` file contains all the optimal hyperparameter configurations for every dataset, wrapped in our lambda sweeping logic. 
   
   To start the sweep, pass the GPU device ID (e.g., `0`) as an argument:
   ```bash
   bash run_gnn_sweep.sh 0
   ```

### What to expect during the run

For **every single dataset and model**, the script will execute 6 consecutive experiments:
- **Run 1 (Baseline):** Training with `lambda=0` (Standard loss).
- **Runs 2-6 (Regularized):** Training with `--use_reg` enabled and `--lambda_val` set to `0.1, 0.5, 1.0, 2.0, 5.0` respectively.

### Checking the Results
You do not need to monitor the console. All results are continuously saved.
- Navigate to the `results/[dataset]/MPNN_[gnn].csv` files.
- Each line will show the hyperparameters, the regularization status (e.g. `REG: 0.5`), and the mean test accuracy ± standard deviation.

> [!WARNING]
> Keep an eye out for `CUDA Out Of Memory` errors when the script hits the `Coauthor-Physics` or `Coauthor-CS` datasets, as those might push past the 4GB limit depending on the exact PyTorch environment overhead. If they crash, the script will simply skip them and continue to the next dataset.

When you're ready to proceed with the Graph Transformers (Phase 2), we will dive into modifying their tuning loops!
