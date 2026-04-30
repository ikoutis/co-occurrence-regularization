import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from lg_parse import parse_method, parser_add_main_args
import sys

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from eval import *
from regularization import estimate_cooccurrence_matrix, edge_loss


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])

# Keep original edges (no self-loops) for co-occurrence regularization
reg_edge_index = dataset.graph['edge_index'].to(device)

dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Penalty matrix construction ###
penalty_matrix = None
if args.use_reg and args.oracle_reg:
    print("Computing oracle penalty matrix from true labels...")
    true_probs = F.one_hot(dataset.label.squeeze(1), c).float()
    co_matrix = estimate_cooccurrence_matrix(true_probs, reg_edge_index, c, device)
    penalty_matrix = -torch.log(co_matrix + 1e-6)
    print("Oracle penalty matrix frozen.")

if args.use_reg and args.mlp_reg:
    print(f"Pre-training MLP for {args.mlp_epochs} epochs to estimate co-occurrence matrix...")
    mlp = nn.Sequential(
        nn.Linear(d, args.hidden_channels),
        nn.ReLU(),
        nn.Linear(args.hidden_channels, c)
    ).to(device)
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
    train_idx = split_idx_lst[0]['train'].to(device)
    for ep in range(args.mlp_epochs):
        mlp.train()
        mlp_opt.zero_grad()
        out = F.log_softmax(mlp(dataset.graph['node_feat']), dim=1)
        loss = F.nll_loss(out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        mlp_opt.step()
        if (ep + 1) % 100 == 0:
            print(f"  MLP epoch {ep+1}/{args.mlp_epochs}, loss={loss.item():.4f}")
    mlp.eval()
    with torch.no_grad():
        mlp_probs = torch.softmax(mlp(dataset.graph['node_feat']), dim=1)
        co_matrix = estimate_cooccurrence_matrix(mlp_probs, reg_edge_index, c, device)
        penalty_matrix = -torch.log(co_matrix + 1e-6)
    print("MLP co-occurrence penalty matrix computed and frozen.")

### Load method ###
model = parse_method(args, n, c, d, device)

criterion = nn.NLLLoss()
eval_func = eval_acc
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')
    if args.save_model:
        save_model(args, model, optimizer, run)

    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad()

        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        out = F.log_softmax(out, dim=1)
        loss = criterion(
            out[train_idx], dataset.label.squeeze(1)[train_idx])

        if args.use_reg and penalty_matrix is not None:
            node_probs = torch.exp(out)
            reg = edge_loss(node_probs, reg_edge_index, penalty_matrix)
            scale = loss.detach() / (reg.detach().abs() + 1e-8)
            loss = loss + args.lambda_val * scale * reg

        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            if args.save_model:
                save_model(args, model, optimizer, run)

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * best_test:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()
### Save results ###
save_result(args, results)
