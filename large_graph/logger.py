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
def save_model(args, model, optimizer, run):
    if not os.path.exists(f'models/{args.dataset}'):
        os.makedirs(f'models/{args.dataset}')
    if(args.model=='MPNN'):
        model_path = f'models/{args.dataset}/{args.model}_{run}.pt'
    else:
        model_path = f'models/{args.dataset}/{args.model}_{run}.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_path)

def load_model(args, model, optimizer, run):
    if(args.model=='MPNN'):
        model_path = f'models/{args.dataset}/{args.model}_{run}.pt'
    else:
        model_path = f'models/{args.dataset}/{args.model}_{run}.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def save_result(args, results):
    result_dir = getattr(args, 'result_dir', 'results')
    out_dir = f'{result_dir}/{args.dataset}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = f'{out_dir}/{args.model}.csv'
    if getattr(args, 'sage', False):
        filename = f'{out_dir}/MPNN_sage.csv'
    else:
        filename = f'{out_dir}/MPNN_gcn.csv'
    print(f"Saving results to {filename}")
    if getattr(args, 'use_reg', False) and getattr(args, 'mlp_reg', False):
        reg_tag = f"MLP_REG: {args.lambda_val}"
    elif getattr(args, 'use_reg', False) and getattr(args, 'oracle_reg', False):
        reg_tag = f"ORACLE_REG: {args.lambda_val}"
    elif not getattr(args, 'use_reg', False):
        reg_tag = "REG: False"
    else:
        reg_tag = f"REG: {args.lambda_val}"
    with open(filename, 'a+') as write_obj:
        write_obj.write(
            f"MPNN {args.lr} {args.hidden_channels} {args.local_layers} {args.dropout} "
            f"{args.ln} {args.bn} {args.res} {reg_tag} "
            f"{results.mean():.2f} $\\pm$ {results.std():.2f} \n")

