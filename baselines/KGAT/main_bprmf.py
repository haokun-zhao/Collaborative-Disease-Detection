import os
import sys
import random
import tracemalloc
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.BPRMF import BPRMF
from parser.parser_bprmf import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_bprmf import DataLoaderBPRMF


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


def evaluate(model, dataloader, Ks, device):
    Ks = [int(k) for k in Ks]
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg', 'hit rate', 'mrr', 'AUC']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, is_train=False)       # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            stacked = np.concatenate(metrics_dict[k][m])
            if m == 'AUC':
                metrics_dict[k][m] = float(np.nanmean(stacked))
            else:
                metrics_dict[k][m] = float(np.mean(stacked))
    return cf_scores, metrics_dict


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
    data = DataLoaderBPRMF(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = BPRMF(args, data.n_users, data.n_items, user_pre_embed, item_pre_embed)
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
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {
        k: {'precision': [], 'recall': [], 'ndcg': [], 'hit rate': [], 'mrr': [], 'AUC': []}
        for k in Ks
    }

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
    for epoch in range(1, args.n_epoch + 1):
        model.train()

        # train cf
        time1 = time()
        total_loss = 0
        n_batch = data.n_cf_train // data.train_batch_size + 1

        for iter in range(1, n_batch + 1):
            time2 = time()
            batch_user, batch_pos_item, batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.train_batch_size)
            batch_user = batch_user.to(device)
            batch_pos_item = batch_pos_item.to(device)
            batch_neg_item = batch_neg_item.to(device)
            batch_loss = model(batch_user, batch_pos_item, batch_neg_item, is_train=True)

            if np.isnan(batch_loss.cpu().detach().numpy()):
                logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

            if (iter % args.print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_batch, time() - time2, batch_loss.item(), total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Total Loss {:.4f}'.format(epoch, n_batch, time() - time1, total_loss))
        epoch_times.append(time() - time1)

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time3 = time()
            _, metrics_dict = evaluate(model, data, Ks, device)
            eval_elapsed = time() - time3
            eval_times.append(eval_elapsed)
            auc_v = metrics_dict[Ks[0]]['AUC']
            auc_log = (
                'N/A' if (isinstance(auc_v, float) and np.isnan(auc_v))
                else _fmt_metric_over_k(metrics_dict, Ks, 'AUC')
            )
            logging.info(
                'CF Evaluation: Epoch {:04d} | Time {:.1f}s | Ks={} | Precision {}  Recall {}  NDCG {}  Hit {}  MRR {}  AUC {}'.format(
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
    data = DataLoaderBPRMF(args, logging)

    # load model
    model = BPRMF(args, data.n_users, data.n_items)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = [int(k) for k in eval(args.Ks)]

    cf_scores, metrics_dict = evaluate(model, data, Ks, device)
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
    args = parse_bprmf_args()
    train(args)
    # predict(args)


