import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from eval import *
from parse import parse_method, parser_add_main_args
from regularization import estimate_cooccurrence_matrix, edge_loss
from model import MLP

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

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)


dataset.label = dataset.label.to(device)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
model = parse_method(args, n, c, d, device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

args.method = args.gnn
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')
    if args.save_model:
        save_model(args, model, optimizer, run)

    penalty_matrix = None

    if args.use_reg and getattr(args, 'oracle_reg', False):
        print("Computing oracle penalty matrix from true labels...")
        if args.dataset == 'questions' and dataset.label.shape[1] > 1:
            true_probs = dataset.label.float()
        else:
            true_probs = F.one_hot(dataset.label.squeeze(1), c).float()
        co_matrix = estimate_cooccurrence_matrix(true_probs, dataset.graph['edge_index'], c, device)
        penalty_matrix = -torch.log(co_matrix + 1e-6)
        print("Oracle penalty matrix frozen.")

    if args.use_reg and getattr(args, 'mlp_reg', False):
        print(f"Pre-training MLP for {args.mlp_epochs} epochs to generate co-occurrence matrix...")
        mlp = MLP(d, args.hidden_channels, c, num_layers=max(2, args.local_layers), dropout=args.dropout).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        
        for mlp_epoch in range(args.mlp_epochs):
            mlp.train()
            mlp_optimizer.zero_grad()
            out = mlp(dataset.graph['node_feat'])
            
            if args.dataset in ('questions'):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                else:
                    true_label = dataset.label
                loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
            else:
                out_log_softmax = F.log_softmax(out, dim=1)
                loss = criterion(out_log_softmax[train_idx], dataset.label.squeeze(1)[train_idx])
            loss.backward()
            mlp_optimizer.step()
            
        with torch.no_grad():
            mlp.eval()
            current_out = mlp(dataset.graph['node_feat'])
            preds = torch.sigmoid(current_out) if args.dataset == 'questions' else torch.exp(F.log_softmax(current_out, dim=1))
            co_matrix = estimate_cooccurrence_matrix(preds, dataset.graph['edge_index'], c, device)
            penalty_matrix = -torch.log(co_matrix + 1e-6)
        print("MLP pre-training complete. Penalty matrix frozen.")

    for epoch in range(args.epochs):
        
        if args.use_reg and not getattr(args, 'mlp_reg', False) and not getattr(args, 'oracle_reg', False) and epoch >= args.reg_start_epoch and (epoch - args.reg_start_epoch) % args.reg_update_freq == 0:
            with torch.no_grad():
                model.eval()
                current_out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
                preds = torch.sigmoid(current_out) if args.dataset == 'questions' else torch.exp(F.log_softmax(current_out, dim=1))
                co_matrix = estimate_cooccurrence_matrix(preds, dataset.graph['edge_index'], c, device)
                penalty_matrix = -torch.log(co_matrix + 1e-6)

        model.train()
        optimizer.zero_grad()

        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        if args.dataset in ('questions'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
                
        if args.use_reg and penalty_matrix is not None:
            if args.dataset == 'questions':
                node_probs = torch.sigmoid(out)
            else:
                node_probs = torch.exp(out)
            reg_loss = edge_loss(node_probs, dataset.graph['edge_index'], penalty_matrix)
            scale = loss.detach() / (reg_loss.detach().abs() + 1e-8)
            loss = loss + args.lambda_val * scale * reg_loss
                
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

