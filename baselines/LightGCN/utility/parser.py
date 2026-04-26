import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run LightGCN.")
    parser.add_argument('--weights_path', nargs='?', default='model/',
                        help='Directory to store checkpoints.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Root data path.')
    parser.add_argument('--dataset', nargs='?', default='mimicIV',
                        help='Dataset: mimicIV | eICU')

    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1,
                        help='Print every N epochs (0 = silent).')
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--embed_size', type=int, default=64)
    # layer_size controls the number of LightGCN layers (depth = len(list))
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='List length = number of propagation layers.')
    parser.add_argument('--batch_size', type=int, default=1024)

    parser.add_argument('--regs', nargs='?', default='[1e-4]',
                        help='L2 regularisation coefficient.')
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Adjacency type: plain | norm | mean')
    parser.add_argument('--gpu_id', type=int, default=0)

    # dropout flags kept for API compatibility (LightGCN uses no dropout by default)
    parser.add_argument('--node_dropout_flag', type=int, default=0)
    parser.add_argument('--node_dropout', nargs='?', default='[0.0]')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.0,0.0,0.0]')

    parser.add_argument('--Ks', nargs='?', default='[3,5,10,20]',
                        help='Top-K values for evaluation.')
    parser.add_argument('--save_flag', type=int, default=1)
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='part | full')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--seeds', nargs='?', default='[42,123,456]',
                        help='List of seeds to run sequentially (used by run_all).')
    parser.add_argument('--run_all_seeds', type=int, default=0,
                        help='If 1, iterate over --seeds automatically.')

    return parser.parse_args()
