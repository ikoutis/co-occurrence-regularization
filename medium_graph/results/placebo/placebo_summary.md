# Placebo experiment summary

λ selected by mean validation accuracy; test reported at λ*.
penalty_dist ≈ 0 ⇒ the transform barely changed the penalty and that placebo row has no power.

| dataset        | model     | condition         |   lambda* |   n_pairs | baseline     | test         |   paired_Δ |   p (Wilcoxon) |   penalty_dist |   offdiag_cv |
|:---------------|:----------|:------------------|----------:|----------:|:-------------|:-------------|-----------:|---------------:|---------------:|-------------:|
| amazon-ratings | MPNN_gcn  | mlp/homophily     |      0.4  |        10 | 53.46 ± 0.63 | 53.81 ± 0.61 |       0.34 |          0.127 |          0.389 |        0.447 |
| amazon-ratings | MPNN_gcn  | mlp/none          |      0.4  |        10 | 53.46 ± 0.63 | 54.09 ± 0.40 |       0.63 |          0.006 |          0     |        0.447 |
| amazon-ratings | MPNN_gcn  | mlp/none/mlp_ep10 |      0.4  |        10 | 53.46 ± 0.63 | 53.74 ± 0.53 |       0.28 |          0.275 |          0     |        0.279 |
| amazon-ratings | MPNN_gcn  | mlp/none/mlp_ep50 |      0.4  |        10 | 53.46 ± 0.63 | 53.95 ± 0.44 |       0.49 |          0.025 |          0     |        0.496 |
| amazon-ratings | MPNN_gcn  | mlp/shuffle       |      0.4  |        10 | 53.46 ± 0.63 | 53.47 ± 0.52 |       0    |          0.557 |          0.596 |        0.447 |
| amazon-ratings | MPNN_gcn  | oracle/none       |      0.4  |        10 | 53.46 ± 0.63 | 53.88 ± 0.52 |       0.42 |          0.064 |          0     |        0.411 |
| amazon-ratings | MPNN_sage | mlp/homophily     |      0.4  |        10 | 55.19 ± 0.61 | 55.50 ± 0.57 |       0.3  |          0.105 |          0.383 |        0.459 |
| amazon-ratings | MPNN_sage | mlp/none          |      0.4  |        10 | 55.19 ± 0.61 | 55.64 ± 0.53 |       0.44 |          0.02  |          0     |        0.459 |
| amazon-ratings | MPNN_sage | mlp/none/mlp_ep10 |      0.05 |        10 | 55.19 ± 0.61 | 55.25 ± 0.59 |       0.06 |          0.77  |          0     |        0.246 |
| amazon-ratings | MPNN_sage | mlp/none/mlp_ep50 |      0.4  |        10 | 55.19 ± 0.61 | 55.43 ± 0.39 |       0.23 |          0.074 |          0     |        0.441 |
| amazon-ratings | MPNN_sage | mlp/shuffle       |      0.1  |        10 | 55.19 ± 0.61 | 55.22 ± 0.47 |       0.03 |          1     |          0.599 |        0.459 |
| amazon-ratings | MPNN_sage | oracle/none       |      0.4  |        10 | 55.19 ± 0.61 | 55.62 ± 0.47 |       0.42 |          0.014 |          0     |        0.411 |
| citeseer       | MPNN_gat  | mlp/homophily     |      0.2  |        10 | 71.86 ± 0.58 | 71.92 ± 0.48 |       0.06 |          0.291 |          0.048 |        0.055 |
| citeseer       | MPNN_gat  | mlp/none          |      0.2  |        10 | 71.86 ± 0.58 | 71.86 ± 0.53 |       0    |          1     |          0     |        0.055 |
| citeseer       | MPNN_gat  | mlp/none/mlp_ep10 |      0.01 |        10 | 71.86 ± 0.58 | 71.71 ± 0.54 |      -0.15 |          0.793 |          0     |        0.045 |
| citeseer       | MPNN_gat  | mlp/none/mlp_ep50 |      0.4  |        10 | 71.86 ± 0.58 | 71.93 ± 0.69 |       0.07 |          0.477 |          0     |        0.082 |
| citeseer       | MPNN_gat  | mlp/shuffle       |      0.2  |        10 | 71.86 ± 0.58 | 71.88 ± 0.56 |       0.02 |          0.865 |          0.116 |        0.055 |
| citeseer       | MPNN_gat  | oracle/none       |      0.4  |        10 | 71.86 ± 0.58 | 73.50 ± 0.48 |       1.64 |          0.004 |          0     |        0.207 |
| cora           | MPNN_gat  | mlp/homophily     |      0.4  |        10 | 83.01 ± 1.10 | 83.69 ± 0.65 |       0.68 |          0.1   |          0.079 |        0.09  |
| cora           | MPNN_gat  | mlp/none          |      0.4  |        10 | 83.01 ± 1.10 | 83.08 ± 1.08 |       0.07 |          0.719 |          0     |        0.09  |
| cora           | MPNN_gat  | mlp/none/mlp_ep10 |      0.4  |        10 | 83.01 ± 1.10 | 83.32 ± 0.87 |       0.31 |          0.215 |          0     |        0.099 |
| cora           | MPNN_gat  | mlp/none/mlp_ep50 |      0.4  |        10 | 83.01 ± 1.10 | 83.43 ± 0.83 |       0.42 |          0.316 |          0     |        0.179 |
| cora           | MPNN_gat  | mlp/shuffle       |      0.01 |        10 | 83.01 ± 1.10 | 83.72 ± 0.62 |       0.71 |          0.016 |          0.243 |        0.09  |
| cora           | MPNN_gat  | oracle/none       |      0.4  |        10 | 83.01 ± 1.10 | 82.99 ± 0.71 |      -0.02 |          0.934 |          0     |        0.303 |
| cora           | MPNN_gcn  | mlp/homophily     |      0.4  |        10 | 84.38 ± 0.62 | 84.51 ± 0.52 |       0.13 |          0.643 |          0.069 |        0.078 |
| cora           | MPNN_gcn  | mlp/none          |      0.4  |        10 | 84.38 ± 0.62 | 84.71 ± 0.59 |       0.33 |          0.102 |          0     |        0.078 |
| cora           | MPNN_gcn  | mlp/none/mlp_ep10 |      0.4  |        10 | 84.38 ± 0.62 | 84.40 ± 0.57 |       0.02 |          0.854 |          0     |        0.032 |
| cora           | MPNN_gcn  | mlp/none/mlp_ep50 |      0.4  |        10 | 84.38 ± 0.62 | 84.79 ± 0.53 |       0.41 |          0.08  |          0     |        0.103 |
| cora           | MPNN_gcn  | mlp/shuffle       |      0.01 |        10 | 84.38 ± 0.62 | 84.33 ± 0.63 |      -0.05 |          0.445 |          0.278 |        0.078 |
| cora           | MPNN_gcn  | oracle/none       |      0.2  |        10 | 84.38 ± 0.62 | 84.85 ± 0.53 |       0.47 |          0.07  |          0     |        0.303 |
| roman-empire   | MPNN_gat  | mlp/homophily     |      0.1  |        10 | 90.29 ± 0.51 | 90.25 ± 0.54 |      -0.04 |          0.846 |          0.264 |        0.257 |
| roman-empire   | MPNN_gat  | mlp/none          |      0.05 |        10 | 90.29 ± 0.51 | 90.31 ± 0.48 |       0.02 |          0.77  |          0     |        0.257 |
| roman-empire   | MPNN_gat  | mlp/none/mlp_ep10 |      0.4  |        10 | 90.29 ± 0.51 | 90.23 ± 0.51 |      -0.05 |          0.695 |          0     |        0.178 |
| roman-empire   | MPNN_gat  | mlp/none/mlp_ep50 |      0.01 |        10 | 90.29 ± 0.51 | 90.31 ± 0.40 |       0.03 |          0.826 |          0     |        0.259 |
| roman-empire   | MPNN_gat  | mlp/shuffle       |      0.01 |        10 | 90.29 ± 0.51 | 90.34 ± 0.52 |       0.06 |          0.846 |          0.392 |        0.257 |
| roman-empire   | MPNN_gat  | oracle/none       |      0.1  |        10 | 90.29 ± 0.51 | 90.24 ± 0.53 |      -0.04 |          0.77  |          0     |        0.605 |
| squirrel       | MPNN_gat  | mlp/homophily     |      0.05 |        10 | 40.23 ± 2.36 | 40.01 ± 2.19 |      -0.22 |          0.922 |          0.188 |        0.224 |
| squirrel       | MPNN_gat  | mlp/none          |      0.01 |        10 | 40.23 ± 2.36 | 40.92 ± 1.66 |       0.69 |          0.492 |          0     |        0.224 |
| squirrel       | MPNN_gat  | mlp/none/mlp_ep10 |      0.01 |        10 | 40.23 ± 2.36 | 40.16 ± 2.38 |      -0.06 |          0.922 |          0     |        0.544 |
| squirrel       | MPNN_gat  | mlp/none/mlp_ep50 |      0.01 |        10 | 40.23 ± 2.36 | 40.48 ± 2.44 |       0.25 |          0.695 |          0     |        0.299 |
| squirrel       | MPNN_gat  | mlp/shuffle       |      0.01 |        10 | 40.23 ± 2.36 | 40.52 ± 2.45 |       0.29 |          0.695 |          0.308 |        0.224 |
| squirrel       | MPNN_gat  | oracle/none       |      0.05 |        10 | 40.23 ± 2.36 | 40.69 ± 1.98 |       0.46 |          0.496 |          0     |        0.044 |
| squirrel       | MPNN_gcn  | mlp/homophily     |      0.01 |        10 | 44.41 ± 2.47 | 43.65 ± 2.49 |      -0.76 |          0.129 |          0.194 |        0.23  |
| squirrel       | MPNN_gcn  | mlp/none          |      0.01 |        10 | 44.41 ± 2.47 | 44.11 ± 2.57 |      -0.3  |          0.492 |          0     |        0.23  |
| squirrel       | MPNN_gcn  | mlp/none/mlp_ep10 |      0.01 |        10 | 44.41 ± 2.47 | 44.11 ± 2.20 |      -0.31 |          0.609 |          0     |        0.369 |
| squirrel       | MPNN_gcn  | mlp/none/mlp_ep50 |      0.01 |        10 | 44.41 ± 2.47 | 43.93 ± 2.64 |      -0.49 |          0.492 |          0     |        0.283 |
| squirrel       | MPNN_gcn  | mlp/shuffle       |      0.1  |        10 | 44.41 ± 2.47 | 44.31 ± 2.47 |      -0.1  |          1     |          0.313 |        0.23  |
| squirrel       | MPNN_gcn  | oracle/none       |      0.01 |        10 | 44.41 ± 2.47 | 44.09 ± 2.44 |      -0.32 |          0.385 |          0     |        0.044 |
