from model import MPNNs, MLP

def parse_method(args, n, c, d, device):
    if args.model == 'MLP':
        model = MLP(d, args.hidden_channels, c,
                    num_layers=args.local_layers, dropout=args.dropout).to(device)
    else:
        model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout,
                      heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear,
                      res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn=args.gnn).to(device)
    return model
        

def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    parser.add_argument('--model', type=str, default='MPNN')
    parser.add_argument('--result_dir', type=str, default='results')
    # GNN
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')
    
    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='models', help='directory for model checkpoints (isolate per SLURM task to avoid collisions)')

    # regularization
    parser.add_argument('--use_reg', action='store_true', help='enable dynamic structure-aware regularization')
    parser.add_argument('--lambda_val', type=float, default=0.5, help='regularization weight')
    parser.add_argument('--reg_start_epoch', type=int, default=10, help='epoch to start applying regularization')
    parser.add_argument('--reg_update_freq', type=int, default=5, help='how often to update the penalty matrix')
    
    # mlp inference variant
    parser.add_argument('--mlp_reg', action='store_true', help='enable static structure-aware regularization from pre-trained MLP')
    parser.add_argument('--mlp_epochs', type=int, default=500, help='number of epochs to pre-train MLP')

    # oracle variant: penalty matrix computed from true labels (upper-bound experiment)
    parser.add_argument('--oracle_reg', action='store_true', help='enable oracle regularization (penalty matrix from true labels)')

    # placebo / ablation controls
    parser.add_argument('--penalty_transform', type=str, default='none',
                        choices=['none', 'shuffle', 'homophily'],
                        help='ablation transform applied to the co-occurrence matrix: '
                             'shuffle destroys all class semantics (placebo), '
                             'homophily keeps the diagonal but removes class-pair structure')
    parser.add_argument('--paired_seeds', action='store_true',
                        help='re-seed model init and training per run so baseline and '
                             'regularized runs are paired (enables paired statistics)')

    # low-label experiment (E3.1) — see LOWLABEL_README.md
    parser.add_argument('--count_reg', action='store_true',
                        help='penalty from co-occurrence COUNTED on train-train edges '
                             '(leakage-free estimator; needs true labels of train nodes only)')
    parser.add_argument('--count_smoothing', type=float, default=1.0,
                        help='additive smoothing for the count estimator')
    parser.add_argument('--train_per_class', type=int, default=0,
                        help='if >0, subsample this run\'s train split to N labeled nodes '
                             'per class (deterministic from the run seed; use for '
                             'fixed-split datasets)')
    parser.add_argument('--valid_per_class', type=int, default=0,
                        help='if >0, subsample the validation split to N nodes per class '
                             '(matched-budget validation sensitivity check)')
    parser.add_argument('--resample_split_per_run', action='store_true',
                        help='with --rand_split_class: draw a fresh label sample each run '
                             '(seeded by seed+run, so conditions stay paired) instead of '
                             'one fixed sample for all runs')
    parser.add_argument('--budget_tag', type=str, default='',
                        help='free-form label budget tag written to the runs CSV')

    # transfer experiment (E4) — see TRANSFER_README.md
    parser.add_argument('--cooc_file', type=str, default='',
                        help='penalty from a PRECOMPUTED row-normalized co-occurrence '
                             'matrix (torch.save; see make_cooc_prior.py) — the '
                             'legitimate-source condition')
    parser.add_argument('--year_split', type=int, default=0,
                        help='if >0, draw train/valid/test only from nodes with '
                             'node_year >= this (target pool); nodes before it are '
                             'the source pool (temporal graphs such as ogbn-arxiv)')
    parser.add_argument('--source_mode', type=str, default='keep',
                        choices=['keep', 'drop', 'labels'],
                        help='with --year_split: keep source nodes unlabeled, drop '
                             'them (induced target subgraph), or add their labels '
                             'to the training set (reference)')
