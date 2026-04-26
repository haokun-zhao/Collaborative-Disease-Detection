import sys
import random
import itertools
import tracemalloc
from time import time

import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from model.NFM import NFM
from parser.parser_nfm import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_nfm import DataLoaderNFM


def _fmt_metric_over_k(metrics_dict, Ks, key):
    parts = []
    for k in Ks:
        v = metrics_dict[k][key]
        if isinstance(v, float) and np.isnan(v):
            parts.append('nan')
        else:
            parts.append('%.4f' % float(v))
    return '[' + ', '.join(parts) + ']'


def _fmt_best_over_k(best_row, Ks, mkey):
    parts = []
    col_base = mkey.replace(' ', '_')
    for kk in Ks:
        v = best_row['{}@{}'.format(col_base, kk)]
        if isinstance(v, float) and np.isnan(v):
            parts.append('nan')
        else:
            parts.append('%.4f' % float(v))
    return '[' + ', '.join(parts) + ']'


def evaluate_batch(model, dataloader, user_ids, Ks):
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    n_users = len(user_ids)
    n_items = dataloader.n_items
    item_ids = list(range(n_items))
    user_idx_map = dict(zip(user_ids, range(n_users)))

    feature_values = dataloader.generate_test_batch(user_ids)
    with torch.no_grad():
        scores = model(feature_values, is_train=False)              # (batch_size)

    rows = [user_idx_map[u] for u in np.repeat(user_ids, n_items).tolist()]
    cols = item_ids * n_users
    score_matrix = torch.Tensor(sp.coo_matrix((scores, (rows, cols)), shape=(n_users, n_items)).todense())

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    metrics_dict = calc_metrics_at_k(score_matrix, train_user_dict, test_user_dict, user_ids, item_ids, Ks)

    score_matrix = score_matrix.numpy()
    return score_matrix, metrics_dict


def evaluate_mp(model, dataloader, Ks, num_processes, device):
    test_batch_size = dataloader.test_batch_size
    test_user_dict = dataloader.test_user_dict

    model.eval()
    model.to("cpu")

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    pool = mp.Pool(num_processes)
    res = pool.starmap(evaluate_batch, [(model, dataloader, batch_user, Ks) for batch_user in user_ids_batches])
    pool.close()

    score_matrix = np.concatenate([r[0] for r in res], axis=0)
    metrics_dict = {k: {} for k in Ks}
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg', 'hit rate', 'mrr', 'AUC']:
            conc = np.concatenate([r[1][k][m] for r in res])
            metrics_dict[k][m] = np.nanmean(conc) if m == 'AUC' else conc.mean()

    torch.cuda.empty_cache()
    model.to(device)
    return score_matrix, metrics_dict


def evaluate(model, dataloader, Ks, num_processes, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    n_users = len(user_ids)
    n_items = dataloader.n_items
    item_ids = list(range(n_items))
    user_idx_map = dict(zip(user_ids, range(n_users)))

    cf_users = []
    cf_items = []
    cf_scores = []

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user in user_ids_batches:
            feature_values = dataloader.generate_test_batch(batch_user)
            feature_values = feature_values.to(device)

            with torch.no_grad():
                batch_scores = model(feature_values, is_train=False)            # (batch_size)

            cf_users.extend(np.repeat(batch_user, n_items).tolist())
            cf_items.extend(item_ids * len(batch_user))
            cf_scores.append(batch_scores.cpu())
            pbar.update(1)

    rows = [user_idx_map[u] for u in cf_users]
    cols = cf_items
    cf_scores = torch.cat(cf_scores)
    cf_score_matrix = torch.Tensor(sp.coo_matrix((cf_scores, (rows, cols)), shape=(n_users, n_items)).todense())

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    metrics_dict = calc_metrics_at_k(cf_score_matrix, train_user_dict, test_user_dict, user_ids, item_ids, Ks)

    cf_score_matrix = cf_score_matrix.numpy()
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg', 'hit rate', 'mrr', 'AUC']:
            value = metrics_dict[k][m]
            if m == 'AUC':
                metrics_dict[k][m] = float(np.nanmean(value)) if hasattr(value, '__len__') else float(value)
            else:
                metrics_dict[k][m] = value.mean() if hasattr(value, 'mean') else float(value)
    return cf_score_matrix, metrics_dict


def evaluate_metrics_only(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    n_items = dataloader.n_items
    item_ids = list(range(n_items))

    batch_metrics = []
    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user in user_ids_batches:
            feature_values = dataloader.generate_test_batch(batch_user)
            feature_values = feature_values.to(device)

            with torch.no_grad():
                batch_scores = model(feature_values, is_train=False)

            # Build dense scores only for this batch to bound memory
            rows = list(range(len(batch_user)))
            rows = [r for r in np.repeat(rows, n_items).tolist()]
            cols = item_ids * len(batch_user)
            score_matrix = torch.Tensor(sp.coo_matrix((batch_scores.cpu(), (rows, cols)), shape=(len(batch_user), n_items)).todense())

            batch_user_arr = np.array(batch_user)
            item_ids_arr = np.array(item_ids)
            metrics = calc_metrics_at_k(score_matrix, train_user_dict, test_user_dict, batch_user_arr, item_ids_arr, Ks)
            batch_metrics.append(metrics)
            pbar.update(1)

    metrics_dict = {k: {} for k in Ks}
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg', 'hit rate', 'mrr', 'AUC']:
            v = np.concatenate([np.atleast_1d(bm[k][m]) for bm in batch_metrics])
            metrics_dict[k][m] = np.nanmean(v) if m == 'AUC' else v.mean()

    # Return empty scores to match call signature; caller in training ignores it
    return np.empty((0, 0)), metrics_dict


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderNFM(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = NFM(args, data.n_users, data.n_items, data.n_entities, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = [int(k) for k in eval(args.Ks)]
    k_min = min(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': [], 'hit rate': [], 'mrr': [], 'AUC': []} for k in Ks}

    num_processes = args.test_cores
    if num_processes and num_processes > 1:
        evaluate_func = evaluate_mp
    else:
        # Use memory-safe evaluator during training to avoid building full dense matrix
        evaluate_func = lambda model, data, Ks, _np, device: evaluate_metrics_only(model, data, Ks, device)

    # efficiency tracking
    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    train_start  = time()
    epoch_times  = []
    eval_times   = []
    conv_time    = None
    n_test_users = len(data.test_user_dict)

    # train model
    for epoch in range(101, args.n_epoch + 1):
        model.train()

        # train cf
        time1 = time()
        total_loss = 0
        n_batch = data.n_cf_train // data.train_batch_size + 1

        for iter in range(1, n_batch + 1):
            pos_feature_values, neg_feature_values = data.generate_train_batch(data.train_user_dict)
            pos_feature_values = pos_feature_values.to(device)
            neg_feature_values = neg_feature_values.to(device)
            batch_loss = model(pos_feature_values, neg_feature_values, is_train=True)

            if np.isnan(batch_loss.cpu().detach().numpy()):
                logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Total Loss {:.4f}'.format(epoch, n_batch, time() - time1, total_loss))
        epoch_times.append(time() - time1)

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch or epoch == 451:
            time3 = time()
            _, metrics_dict = evaluate_func(model, data, Ks, num_processes, device)
            eval_elapsed = time() - time3
            eval_times.append(eval_elapsed)
            auc_v = metrics_dict[Ks[0]]['AUC']
            auc_log = (
                'N/A' if (isinstance(auc_v, float) and np.isnan(auc_v))
                else _fmt_metric_over_k(metrics_dict, Ks, 'AUC')
            )
            logging.info(
                'CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Ks={} | Precision {}  Recall {}  NDCG {}  Hit {}  MRR {}  AUC {}'.format(
                    epoch, eval_elapsed, str(Ks),
                    _fmt_metric_over_k(metrics_dict, Ks, 'precision'),
                    _fmt_metric_over_k(metrics_dict, Ks, 'recall'),
                    _fmt_metric_over_k(metrics_dict, Ks, 'ndcg'),
                    _fmt_metric_over_k(metrics_dict, Ks, 'hit rate'),
                    _fmt_metric_over_k(metrics_dict, Ks, 'mrr'),
                    auc_log)
            )

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg', 'hit rate', 'mrr', 'AUC']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch
                conv_time  = time()

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg', 'hit rate', 'mrr', 'AUC']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m.replace(' ', '_'), k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    if best_epoch >= 0 and best_epoch in metrics_df['epoch_idx'].values:
        best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
        b_auc = best_metrics['AUC@{}'.format(Ks[0])]
        auc_best = 'N/A' if (isinstance(b_auc, float) and np.isnan(b_auc)) else _fmt_best_over_k(best_metrics, Ks, 'AUC')
        logging.info(
            'Best CF Evaluation: Epoch {:04d} | Ks={} | Precision {}  Recall {}  NDCG {}  Hit {}  MRR {}  AUC {}'.format(
                int(best_metrics['epoch_idx']), str(Ks),
                _fmt_best_over_k(best_metrics, Ks, 'precision'),
                _fmt_best_over_k(best_metrics, Ks, 'recall'),
                _fmt_best_over_k(best_metrics, Ks, 'ndcg'),
                _fmt_best_over_k(best_metrics, Ks, 'hit rate'),
                _fmt_best_over_k(best_metrics, Ks, 'mrr'),
                auc_best)
        )

    # efficiency summary
    peak_ram_mb  = tracemalloc.get_traced_memory()[1] / 1024 ** 2
    tracemalloc.stop()
    peak_gpu_mb  = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if torch.cuda.is_available() else 0.0)
    avg_epoch_s    = float(np.mean(epoch_times))    if epoch_times else 0.0
    avg_throughput = (float(n_test_users) / float(np.mean(eval_times))
                      if eval_times else 0.0)
    time_to_conv   = (conv_time - train_start) if conv_time is not None else (time() - train_start)

    logging.info('=== Efficiency Summary ===')
    logging.info('  Avg epoch time        : {:.2f} s'.format(avg_epoch_s))
    logging.info('  Time to convergence   : {:.1f} s  (best epoch {:d})'.format(time_to_conv, best_epoch))
    logging.info('  Peak RAM              : {:.1f} MB'.format(peak_ram_mb))
    logging.info('  Peak GPU memory       : {:.1f} MB'.format(peak_gpu_mb))
    logging.info('  Inference throughput  : {:.1f} users/s'.format(avg_throughput))

    peak_mem = max(peak_ram_mb, peak_gpu_mb)
    eff_df = pd.DataFrame([{
        'training_time_per_epoch_s':       avg_epoch_s,
        'time_to_convergence_s':           time_to_conv,
        'best_epoch':                      best_epoch,
        'peak_ram_mb':                     peak_ram_mb,
        'peak_gpu_mb':                     peak_gpu_mb,
        'peak_memory_mb':                  peak_mem,
        'inference_throughput_patients_s': avg_throughput,
        'avg_epoch_time_s':                avg_epoch_s,
        'inference_throughput_users_s':    avg_throughput,
    }])
    eff_df.to_csv(args.save_dir + '/efficiency.tsv', sep='\t', index=False)


def predict(args):
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderNFM(args, logging)

    # load model
    model = NFM(args, data.n_users, data.n_items, data.n_entities)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    num_processes = args.test_cores
    if num_processes and num_processes > 1:
        evaluate_func = evaluate_mp
    else:
        evaluate_func = evaluate

    Ks = [int(k) for k in eval(args.Ks)]

    cf_scores, metrics_dict = evaluate_func(model, data, Ks, num_processes, device)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    auc_v = metrics_dict[Ks[0]]['AUC']
    auc_p = 'N/A' if (isinstance(auc_v, float) and np.isnan(auc_v)) else _fmt_metric_over_k(metrics_dict, Ks, 'AUC')
    print(
        'CF Evaluation: Ks={} | Precision {}  Recall {}  NDCG {}  Hit {}  MRR {}  AUC {}'.format(
            str(Ks),
            _fmt_metric_over_k(metrics_dict, Ks, 'precision'),
            _fmt_metric_over_k(metrics_dict, Ks, 'recall'),
            _fmt_metric_over_k(metrics_dict, Ks, 'ndcg'),
            _fmt_metric_over_k(metrics_dict, Ks, 'hit rate'),
            _fmt_metric_over_k(metrics_dict, Ks, 'mrr'),
            auc_p)
    )



if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    args = parse_nfm_args()
    train(args)
    # predict(args)


