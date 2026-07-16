def parse_method(args, n, c, d, device):
    if args.model == "GPS":
        from network.gps_model import GPSModel
        model = GPSModel(args, d, c).to(device)
    elif args.model =="polynormer":
        from polynormer import Polynormer
        # official Polynormer falls back to --dropout when --global_dropout is
        # unset; without this, GlobalAttn gets dropout=None and crashes with
        # F.dropout(p=None) at the first forward after the phase switch
        global_dropout = args.global_dropout if args.global_dropout is not None else args.dropout
        model = Polynormer(d, args.hidden_channels, c, local_layers=args.local_layers, global_layers=args.global_layers,
            in_dropout=args.in_dropout, dropout=args.dropout, global_dropout=global_dropout,
            heads=args.num_heads, beta=args.beta, pre_ln=args.pre_ln).to(device)
    elif args.model == 'nodeformer':
        from nodeformer import NodeFormer
        model = NodeFormer(in_channels=d,
                         hidden_channels=args.hidden_channels,
                         out_channels=c,
                         num_layers=args.global_layers,
                         dropout=args.dropout,
                         num_heads=args.num_heads,
                         use_bn=args.use_bn).to(device)
    elif args.model == 'sgformer':
        from sgformer import SGFormer, GCN
        # use_bn must be passed through: GCN defaults to use_bn=True, but the
        # official SGFormer passes args.use_bn (published cora/citeseer/pubmed
        # configs run the GCN branch WITHOUT BatchNorm)
        gnn = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=args.hidden_channels,
                    num_layers=args.layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn)
        # Official SGFormer separates the transformer depth (ours_layers, here
        # tr_layers) from the GCN backbone depth (--layers). The transformer
        # branch also gets its own dropout (ours_dropout, here tr_dropout).
        tr_dropout = args.tr_dropout if args.tr_dropout is not None else args.dropout
        model = SGFormer(d, args.hidden_channels, c, num_layers=args.tr_layers, alpha=args.alpha, dropout=tr_dropout,
                         num_heads=args.num_heads, use_bn=args.use_bn, use_residual=args.use_residual,
                         use_graph=args.use_graph, use_weight=args.use_weight, use_act=args.use_act,
                         graph_weight=args.graph_weight, gnn=gnn, aggregate=args.aggregate, jk=args.jk).to(device)
    elif args.model == 'exphormer':
        from network.multi_model import MultiModel
        model = MultiModel(args, d, c).to(device)
    elif args.model == 'nagphormer':
        from nagphormer import TransformerModel
        model = TransformerModel(hops=args.hops,
                        n_class=c,
                        input_dim=d,
                        pe_dim = args.hidden_channels,
                        n_layers=args.global_layers,
                        num_heads=args.num_heads,
                        hidden_dim=args.hidden_channels,
                        ffn_dim=args.hidden_channels,
                        dropout_rate=args.dropout,
                        attention_dropout_rate=args.dropout).to(device)
    else:
        from goat import Transformer
        model = Transformer(
            num_nodes=n,
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            global_dim=args.hidden_channels,
            num_layers=args.layers,
            heads=args.num_heads,
            ff_dropout=args.dropout,
            attn_dropout=args.dropout,
            skip=0,
            dist_count_norm=1,
            conv_type='full',
            num_centroids=args.num_centroids,
            no_bn=False,
            norm_type='batch_norm'
        ).to(device)

    return model
        

def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='../data/')
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
    # GPS
    parser.add_argument('--pe_types', type=str, default='RWSE')
    parser.add_argument('--local_gnn_type', type=str, default='GCN')
    parser.add_argument('--layers', type=int, default=1)
    # poly
    parser.add_argument('--global_layers', type=int, default=2,
                        help='number of layers for global attention')
    parser.add_argument('--beta', type=float, default=-1.0,
                        help='Polynormer beta initialization')
    parser.add_argument('--global_dropout', type=float, default=None)
    parser.add_argument('--local_epochs', type=int, default=0,
                        help='Polynormer official two-phase schedule: epochs with local '
                             'attention only (model._global=False)')
    parser.add_argument('--global_epochs', type=int, default=0,
                        help='Polynormer official two-phase schedule: epochs with global '
                             'attention, warm-started from the best local checkpoint. '
                             'When local_epochs+global_epochs > 0, --epochs is ignored.')
    # sgformer
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--use_weight', action='store_true')
    parser.add_argument('--attention', type=str, default='gcn')
    parser.add_argument('--graph_weight', type=float,
                        default=0.8, help='graph weight.')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each trans layer')
    parser.add_argument('--use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--tr_layers', type=int, default=1,
                        help='SGFormer transformer-branch depth (official ours_layers; '
                             'the official configs use 1). --layers is the GCN backbone depth.')
    parser.add_argument('--tr_dropout', type=float, default=None,
                        help='SGFormer transformer-branch dropout (official ours_dropout); '
                             'defaults to --dropout')
    parser.add_argument('--tr_weight_decay', type=float, default=None,
                        help='SGFormer transformer-branch weight decay (official '
                             'ours_weight_decay); enables the two-group optimizer')
    parser.add_argument('--aggregate', type=str, default='add',
                        help='aggregate type, add or cat.')
    # nagphormer
    parser.add_argument('--hops', type=int, default=10)
    # goat
    parser.add_argument('--num_centroids', type=int, default=4096)

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--in_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='models', help='directory for model checkpoints (isolate per SLURM task to avoid collisions)')
    parser.add_argument('--save_result', action='store_true', help='whether to save result')

    # regularization
    parser.add_argument('--use_reg', action='store_true', help='enable dynamic structure-aware regularization')
    parser.add_argument('--lambda_val', type=float, default=0.5, help='regularization weight')
    parser.add_argument('--reg_start_epoch', type=int, default=10, help='epoch to start applying regularization')
    parser.add_argument('--reg_update_freq', type=int, default=5, help='how often to update the penalty matrix')

    # mlp inference variant
    parser.add_argument('--mlp_reg', action='store_true', help='static penalty from pre-trained MLP')
    parser.add_argument('--mlp_epochs', type=int, default=500, help='epochs to pre-train the MLP')

    # oracle variant
    parser.add_argument('--oracle_reg', action='store_true', help='oracle penalty from true labels (upper bound)')

    # placebo / ablation controls (see ../PLACEBO_README.md)
    parser.add_argument('--penalty_transform', type=str, default='none',
                        choices=['none', 'shuffle', 'homophily'],
                        help='ablation transform applied to the co-occurrence matrix')
    parser.add_argument('--paired_seeds', action='store_true',
                        help='re-seed model init and training per run so baseline and '
                             'regularized runs are paired (enables paired statistics)')

    # result directory
    parser.add_argument('--result_dir', type=str, default='results', help='directory to save results')


