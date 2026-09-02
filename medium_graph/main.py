import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits, apply_year_split
from eval import *
from parse import parse_method, parser_add_main_args
from regularization import estimate_cooccurrence_matrix, edge_loss, transform_cooccurrence_matrix, penalty_stats, count_cooccurrence_matrix
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

# E4 transfer experiment: temporal source/target pools (see TRANSFER_README.md)
target_pool, source_idx = None, None
if getattr(args, 'year_split', 0) > 0:
    if not args.rand_split_class:
        raise SystemExit('--year_split requires --rand_split_class (per-class sampling from the target pool)')
    dataset, target_pool, source_idx = apply_year_split(dataset, args.year_split, args.source_mode)

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num, pool=target_pool)]
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

def build_penalty(co_matrix, run, meta):
    """Applies the (optional) ablation transform and logs distinguishability."""
    gen = torch.Generator()
    gen.manual_seed(args.seed + 5000 + run)
    co_t = transform_cooccurrence_matrix(co_matrix, args.penalty_transform, gen)
    penalty_orig = -torch.log(co_matrix + 1e-6)
    penalty = -torch.log(co_t + 1e-6)
    rel_dist, offdiag_cv = penalty_stats(penalty_orig, penalty)
    meta['pdist'] = rel_dist
    meta['pcv'] = offdiag_cv
    if args.penalty_transform != 'none':
        print(f"Penalty transform '{args.penalty_transform}': "
              f"rel Frobenius dist to original = {rel_dist:.4f}, "
              f"off-diag CV of original penalty = {offdiag_cv:.4f}")
    return penalty

def subsample_per_class(idx, label, n_per_class):
    """Deterministically (given the current RNG state) subsample idx to at
    most n_per_class nodes per class. Uses the CPU RNG so the draw is
    machine-independent; with --paired_seeds the same run index yields the
    same subsample in every condition."""
    idx = idx.to(label.device)
    y = label.squeeze(1)[idx]
    keep = []
    for cls in y.unique():
        idx_c = idx[y == cls]
        perm = torch.randperm(idx_c.numel())
        keep.append(idx_c[perm.to(idx_c.device)[:n_per_class]])
    keep = torch.cat(keep)
    return keep.sort().values

### Training loop ###
run_meta = []
for run in range(args.runs):
    run_meta.append({})
    # re-seed FIRST so per-run split resampling below is deterministic from
    # seed+run and identical across conditions (split selection consumes no
    # RNG in the legacy paths, so this reordering does not change them)
    if args.paired_seeds:
        fix_seed(args.seed + run)
    if getattr(args, 'resample_split_per_run', False) and args.rand_split_class:
        split_idx = class_rand_splits(
            dataset.label.cpu(), args.label_num_per_class, args.valid_num, args.test_num,
            pool=target_pool)
    elif args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run % len(split_idx_lst)]
    train_idx = split_idx['train'].to(device)
    if getattr(args, 'train_per_class', 0) > 0:
        train_idx = subsample_per_class(train_idx, dataset.label, args.train_per_class)
        print(f"Label budget: {args.train_per_class}/class -> {train_idx.numel()} train nodes")
    if source_idx is not None and args.source_mode == 'labels':
        # reference condition: the source pool's labels are simply added to
        # the training set (what a practitioner who HAS the source labels,
        # not just their K x K summary, would do)
        train_idx = torch.cat([train_idx, source_idx.to(device)]).unique()
        print(f"Source labels added to training: {train_idx.numel()} train nodes")
    if getattr(args, 'valid_per_class', 0) > 0:
        split_idx = dict(split_idx)
        split_idx['valid'] = subsample_per_class(
            split_idx['valid'], dataset.label, args.valid_per_class)
        print(f"Matched-budget validation: {split_idx['valid'].numel()} valid nodes")
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
        penalty_matrix = build_penalty(co_matrix, run, run_meta[run])
        print("Oracle penalty matrix frozen.")

    if args.use_reg and getattr(args, 'count_reg', False):
        print("Counting co-occurrence on train-train edges (leakage-free estimator)...")
        with torch.no_grad():
            if args.dataset == 'questions' and dataset.label.shape[1] > 1:
                true_probs = dataset.label.float()
            else:
                true_probs = F.one_hot(dataset.label.squeeze(1), c).float()
            train_mask = torch.zeros(n, dtype=torch.bool, device=train_idx.device)
            train_mask[train_idx] = True
            co_matrix, n_prior_edges = count_cooccurrence_matrix(
                true_probs, dataset.graph['edge_index'], train_mask, args.count_smoothing)
            penalty_matrix = build_penalty(co_matrix, run, run_meta[run])
            run_meta[run]['n_prior_edges'] = n_prior_edges
            co_oracle = estimate_cooccurrence_matrix(true_probs, dataset.graph['edge_index'], c, device)
            run_meta[run]['cooc_oracle_dist'] = ((co_matrix - co_oracle).norm()
                                                 / co_oracle.norm().clamp(min=1e-12)).item()
        print(f"Count penalty frozen ({n_prior_edges} train-train edges).")

    if args.use_reg and getattr(args, 'cooc_file', ''):
        # E4: precomputed prior from a legitimate source (make_cooc_prior.py).
        # Nothing about the current split touches it — the only per-run
        # element is the (seeded) placebo shuffle, if requested.
        print(f"Loading precomputed co-occurrence prior from {args.cooc_file} ...")
        with torch.no_grad():
            obj = torch.load(args.cooc_file, map_location='cpu')
            co_matrix = (obj['C'] if isinstance(obj, dict) else obj).float().to(device)
            if co_matrix.shape != (c, c):
                raise SystemExit(f'prior shape {tuple(co_matrix.shape)} != ({c}, {c})')
            penalty_matrix = build_penalty(co_matrix, run, run_meta[run])
            if isinstance(obj, dict) and 'n_edges' in obj.get('meta', {}):
                run_meta[run]['n_prior_edges'] = obj['meta']['n_edges']
            # distance to the all-label matrix of the graph AS TRAINED ON
            # (target-only in --source_mode drop): the temporal drift of the prior
            if args.dataset == 'questions' and dataset.label.shape[1] > 1:
                true_probs = dataset.label.float()
            else:
                true_probs = F.one_hot(dataset.label.squeeze(1), c).float()
            co_oracle = estimate_cooccurrence_matrix(true_probs, dataset.graph['edge_index'], c, device)
            run_meta[run]['cooc_oracle_dist'] = ((co_matrix - co_oracle).norm()
                                                 / co_oracle.norm().clamp(min=1e-12)).item()
        print(f"Transferred prior frozen (rel. dist to this graph's oracle matrix: "
              f"{run_meta[run]['cooc_oracle_dist']:.4f}).")

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
            penalty_matrix = build_penalty(co_matrix, run, run_meta[run])
            # P0.6 covariates: how good is the penalty source? (diagnostics
            # only — the oracle matrix is never used in training here)
            if args.dataset == 'questions' and dataset.label.shape[1] > 1:
                true_probs = dataset.label.float()
            else:
                true_probs = F.one_hot(dataset.label.squeeze(1), c).float()
            co_oracle = estimate_cooccurrence_matrix(true_probs, dataset.graph['edge_index'], c, device)
            run_meta[run]['cooc_oracle_dist'] = ((co_matrix - co_oracle).norm()
                                                 / co_oracle.norm().clamp(min=1e-12)).item()
            if args.dataset != 'questions':
                valid_idx = split_idx['valid']
                mlp_pred = current_out[valid_idx].argmax(dim=1)
                run_meta[run]['mlp_acc'] = (mlp_pred == dataset.label.squeeze(1)[valid_idx]).float().mean().item()
        print("MLP pre-training complete. Penalty matrix frozen.")

    # Re-seed after penalty construction: MLP pre-training consumes RNG state,
    # so without this, dropout/training noise would differ between the baseline
    # and regularized conditions and the runs would not be paired.
    if args.paired_seeds:
        fix_seed(args.seed + 100000 + run)

    for epoch in range(args.epochs):
        
        if args.use_reg and not getattr(args, 'mlp_reg', False) and not getattr(args, 'oracle_reg', False) and not getattr(args, 'count_reg', False) and not getattr(args, 'cooc_file', '') and epoch >= args.reg_start_epoch and (epoch - args.reg_start_epoch) % args.reg_update_freq == 0:
            with torch.no_grad():
                model.eval()
                current_out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
                preds = torch.sigmoid(current_out) if args.dataset == 'questions' else torch.exp(F.log_softmax(current_out, dim=1))
                co_matrix = estimate_cooccurrence_matrix(preds, dataset.graph['edge_index'], c, device)
                penalty_matrix = build_penalty(co_matrix, run, run_meta[run])

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
            scale = loss.detach().abs() / reg_loss.detach().abs().clamp(min=1e-4)
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
save_runs_detail(args, logger, run_meta)

