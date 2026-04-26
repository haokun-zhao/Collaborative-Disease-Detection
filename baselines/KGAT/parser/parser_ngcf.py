import argparse


def parse_ngcf_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")

    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--data_name', nargs='?', default='mimicIV',
                        help='Choose a dataset from {mimicIV, eICU}')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Root data directory.')

    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: pretrain embeddings, 2: pretrain model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth')

    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User/item initial embedding dimension.')
    parser.add_argument('--layer_size', nargs='?', default='[64, 64, 64]',
                        help='Output size of each propagation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability for each propagation layer.')
    parser.add_argument('--l2loss_lambda', type=float, default=1e-5,
                        help='L2 regularisation coefficient on initial embeddings.')

    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size',  type=int, default=10000)
    parser.add_argument('--eval_item_batch_size', type=int, default=4096,
                        help='Score items in chunks (CDD batch_test-style). Reduces peak GPU memory only; '
                             'does not reduce total eval time. Use 0 for one matmul over all items.')
    parser.add_argument('--eval_skip_auc', type=int, default=0,
                        help='1: skip sklearn per-user AUC in eval (faster). '
                             'Set 0 to compute AUC (logged as a number; else AUC shows N/A).')

    parser.add_argument('--lr',            type=float, default=0.0001)
    parser.add_argument('--n_epoch',       type=int,   default=500)
    parser.add_argument('--stopping_steps', type=int,  default=10)

    parser.add_argument('--print_every', type=int, default=0,
                        help='Log every N training batches; 0 = only log at end of each epoch.')
    parser.add_argument('--evaluate_every', type=int, default=10)

    parser.add_argument('--Ks', nargs='?', default='[3, 5, 10, 20]',
                        help='Evaluate at these K values.')

    args = parser.parse_args()

    args.save_dir = (
        'trained_model/NGCF/{}/embed-dim{}_layer-size{}_lr{}_pretrain{}_seed{}/'.format(
            args.data_name, args.embed_dim,
            args.layer_size.replace(' ', ''), args.lr, args.use_pretrain, args.seed
        )
    )
    return args
