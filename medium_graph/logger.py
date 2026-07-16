import torch

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest Test: {result[:, 2].max():.2f}')
            print(f'Chosen epoch: {ind}')
            print(f'Final Train: {result[ind, 0]:.2f}')
            print(f'Final Test: {result[ind, 2]:.2f}')
            self.test=result[ind, 2]
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            self.test=r.mean()
            return best_result[:, 4]

    def output(self,out_path,info):
        with open(out_path,'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')

import os
def _model_path(args, run):
    base = getattr(args, 'model_dir', 'models') or 'models'
    os.makedirs(f'{base}/{args.dataset}', exist_ok=True)
    if args.model == 'MPNN':
        return f'{base}/{args.dataset}/{args.model}_{args.gnn}_{run}.pt'
    return f'{base}/{args.dataset}/{args.model}_{run}.pt'

def save_model(args, model, optimizer, run):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, _model_path(args, run))

def load_model(args, model, optimizer, run):
    checkpoint = torch.load(_model_path(args, run))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def config_string(args):
    """Compact base-hyperparameter fingerprint (excludes reg flags), used to
    detect rows produced under different configs in one result_dir."""
    return ','.join([
        f'lr={args.lr}', f'hid={args.hidden_channels}', f'do={args.dropout}',
        f'wd={args.weight_decay}', f'heads={getattr(args, "num_heads", "")}',
        f'll={args.local_layers}', f'gnn={getattr(args, "gnn", "")}',
        f'ln={args.ln}', f'bn={args.bn}', f'res={args.res}',
        f'jk={args.jk}', f'prelin={args.pre_linear}'])

def save_runs_detail(args, logger, run_meta=None):
    """
    Appends one CSV row per run with the validation accuracy and the test
    accuracy at the best-validation epoch. This is what downstream analysis
    needs for validation-based lambda selection and paired statistics —
    the aggregate mean +/- std written by save_result is not enough.
    Also logs the P0.6 covariates (mlp_acc, cooc_oracle_dist) when the
    penalty source is an MLP.
    """
    import csv
    result_dir = getattr(args, 'result_dir', 'results')
    d = f'{result_dir}/{args.dataset}'
    os.makedirs(d, exist_ok=True)
    name = f'{args.model}_{args.gnn}' if args.model == 'MPNN' else f'{args.model}'
    path = f'{d}/runs_{name}.csv'
    write_header = not os.path.exists(path)

    reg_type = 'none'
    if getattr(args, 'use_reg', False):
        if getattr(args, 'oracle_reg', False):
            reg_type = 'oracle'
        elif getattr(args, 'mlp_reg', False):
            reg_type = 'mlp'
        else:
            reg_type = 'dynamic'

    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['dataset', 'model', 'reg_type', 'penalty_transform',
                        'lambda', 'mlp_epochs', 'seed', 'run', 'best_valid',
                        'test_at_best_valid', 'penalty_dist', 'offdiag_cv',
                        'mlp_acc', 'cooc_oracle_dist', 'config'])
        for run, results in enumerate(logger.results):
            if not results:
                continue
            r = 100 * torch.tensor(results)
            ind = r[:, 1].argmax().item()
            meta = run_meta[run] if run_meta and run < len(run_meta) else {}
            w.writerow([args.dataset, name, reg_type,
                        getattr(args, 'penalty_transform', 'none') if reg_type != 'none' else '',
                        getattr(args, 'lambda_val', 0.0) if reg_type != 'none' else 0.0,
                        getattr(args, 'mlp_epochs', '') if reg_type == 'mlp' else '',
                        args.seed, run,
                        f'{r[ind, 1].item():.4f}', f'{r[ind, 2].item():.4f}',
                        f"{meta.get('pdist', '')}", f"{meta.get('pcv', '')}",
                        f"{meta.get('mlp_acc', '')}", f"{meta.get('cooc_oracle_dist', '')}",
                        config_string(args)])

def save_result(args, results):
    result_dir = getattr(args, 'result_dir', 'results')
    os.makedirs(f'{result_dir}/{args.dataset}', exist_ok=True)
    if args.model == 'MPNN':
        filename = f'{result_dir}/{args.dataset}/{args.model}_{args.gnn}.csv'
    else:
        filename = f'{result_dir}/{args.dataset}/{args.model}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        reg_info = "REG: False "
        if getattr(args, 'use_reg', False):
            if getattr(args, 'oracle_reg', False):
                reg_info = f"ORACLE_REG: {args.lambda_val} "
            elif getattr(args, 'mlp_reg', False):
                reg_info = f"MLP_REG: {args.lambda_val} "
            elif getattr(args, 'reg_start_epoch', 10) >= 500:
                reg_info = f"DELAY_REG: {args.lambda_val} "
            else:
                reg_info = f"REG: {args.lambda_val} "
        if(args.model=='MPNN'):
            write_obj.write(
                f"{args.model} " + f"{args.lr} " + f"{args.hidden_channels} " + f"{args.local_layers} " + f"{args.dropout} " + f"{args.ln} " + \
                f"{args.bn} " + f"{args.res} " + reg_info + \
                f"{results.mean():.2f} $\pm$ {results.std():.2f} \n")
        else:
            write_obj.write(
                f"{args.model} " + f"{args.lr} " + reg_info + \
                f"{results.mean():.2f} $\pm$ {results.std():.2f} \n")

