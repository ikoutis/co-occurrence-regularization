# Regularization Integration Updates

## Dynamic Structure-Aware Regularization
**Date**: 2026-04-20

### Concept
We implemented a dynamic structure-aware regularization that estimates the global co-occurrence matrix of classes dynamically during training, rather than relying on a separate pre-trained MLP or a static assumption of homophily.

### Schedule
We agreed on the following schedule to balance adaptability with training stability:
- **Warm-up (Start Epoch)**: 10 epochs. We wait for the model to produce meaningful representations before trusting its predictions to form a penalty matrix.
- **Update Frequency**: Every 5 epochs. This prevents the target from shifting too rapidly, which could cause training oscillations, while still adapting as the model improves.

### Parameters Added
- `--use_reg`: Boolean flag to turn the regularization on.
- `--lambda_val`: Float weight for the regularization loss (default: 0.5).
- `--reg_start_epoch`: Epoch to begin applying regularization (default: 10).
- `--reg_update_freq`: How often to re-estimate the co-occurrence matrix (default: 5).
