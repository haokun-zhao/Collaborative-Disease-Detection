import os
import sys
import random
import tracemalloc
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
import torch
import torch.optim as optim

from model.NGCF import NGCF
from parser.parser_ngcf import parse_ngcf_args
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_bprmf import DataLoaderBPRMF


def _fmt_metric_over_k(metrics, Ks, key):
    """Format one metric as a bracketed list aligned with Ks (same order as --Ks)."""
    parts = []
    for k in Ks:
        v = metrics[k][key]
        if isinstance(v, float) and np.isnan(v):
            parts.append('nan')
        else:
            parts.append('%.4f' % float(v))
    return '[' + ', '.join(parts) + ']'


# ── adjacency matrix construction ────────────────────────────────────────────

def build_norm_adj(n_users, n_items, train_user_dict):
    """
    Build the normalised bipartite adjacency matrix used by NGCF.

    A = [[0, R], [R^T, 0]]   shape: (n_users+n_items) × (n_users+n_items)
    A_hat = D^{-1/2} A D^{-1/2}

    Returns: scipy.sparse.csr_matrix
    """
    n = n_users + n_items
    rows, cols = [], []
    for uid, iids in train_user_dict.items():
        for iid in iids:
            rows.append(uid);           cols.append(n_users + iid)
            rows.append(n_users + iid); cols.append(uid)

    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # D^{-1/2}
    deg = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    A_hat = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocsr()
    return A_hat


# ── evaluation ───────────────────────────────────────────────────────────────

def _batch_scores_from_embeddings(u_emb, i_emb, batch_uids, n_items, eval_item_batch_size, device):
    """
    batch_uids: LongTensor on device, indices into u_emb.
    Optional item chunking (CDD batch_test-style) to cap peak memory on large catalogs.
    """
    u = u_emb[batch_uids]
    chunk = eval_item_batch_size if eval_item_batch_size and eval_item_batch_size > 0 else n_items
    if chunk >= n_items:
        ic = torch.arange(n_items, dtype=torch.long, device=device)
        return torch.matmul(u, i_emb[ic].t())
    parts = []
    for i_start in range(0, n_items, chunk):
        i_end = min(i_start + chunk, n_items)
        ic = torch.arange(i_start, i_end, dtype=torch.long, device=device)
        parts.append(torch.matmul(u, i_emb[ic].t()))
    return torch.cat(parts, dim=1)


def evaluate(model, dataloader, Ks, device, eval_item_batch_size=0, compute_auc=True):
    train_user_dict = dataloader.train_user_dict
    test_user_dict  = dataloader.test_user_dict
    n_items = dataloader.n_items
    item_ids_np = np.arange(n_items, dtype=np.int64)

    model.eval()
    user_ids = list(test_user_dict.keys())
    user_batches = [
        torch.LongTensor(user_ids[i: i + dataloader.test_batch_size])
        for i in range(0, len(user_ids), dataloader.test_batch_size)
    ]

    metric_names = ['precision', 'recall', 'ndcg', 'hit rate', 'mrr', 'AUC']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
    cf_scores = []

    with torch.no_grad():
        u_emb, i_emb = model.get_propagated_embeddings()

    with tqdm(total=len(user_batches), desc='Evaluating') as pbar:
        for batch_uids in user_batches:
            batch_uids = batch_uids.to(device)
            batch_scores = _batch_scores_from_embeddings(
                u_emb, i_emb, batch_uids, n_items, eval_item_batch_size, device
            )
            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(
                batch_scores, train_user_dict, test_user_dict,
                batch_uids.cpu().numpy(), item_ids_np, Ks,
                compute_auc=compute_auc,
            )
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


# ── training ─────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_id), no_console=False)
    logging.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    data = DataLoaderBPRMF(args, logging)
    n_users, n_items = data.n_users, data.n_items
    logging.info('n_users: %d  n_items: %d  n_train: %d  n_test: %d'
                 % (n_users, n_items, data.n_cf_train, data.n_cf_test))

    # Normalised adjacency
    logging.info('Building normalised adjacency matrix ...')
    t0 = time()
    norm_adj = build_norm_adj(n_users, n_items, data.train_user_dict)
    logging.info('  done in %.1fs  nnz=%d' % (time() - t0, norm_adj.nnz))

    # Pretrained embeddings (optional)
    user_pre_embed = item_pre_embed = None
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)

    # Model
    model = NGCF(args, n_users, n_items, norm_adj, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)
    model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    Ks = eval(args.Ks)
    k_min, k_max = min(Ks), max(Ks)

    best_recall   = 0
    best_epoch    = -1
    epoch_list    = []
    metrics_list  = {
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

    for epoch in range(1, args.n_epoch + 1):
        model.train()
        t1 = time()
        total_loss = 0.0
        n_batch = data.n_cf_train // data.train_batch_size + 1

        for it in range(1, n_batch + 1):
            t2 = time()
            u, pos, neg = data.generate_cf_batch(data.train_user_dict, data.train_batch_size)
            u   = u.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            loss = model(u, (pos, neg), is_train=True)

            if np.isnan(loss.item()):
                logging.info('ERROR: NaN loss at Epoch %04d Iter %04d.' % (epoch, it))
                sys.exit()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if args.print_every > 0 and (it % args.print_every) == 0:
                logging.info(
                    'Train Epoch %04d Iter %04d/%04d | Time %.1fs | '
                    'Iter Loss %.4f | Mean Loss %.4f'
                    % (epoch, it, n_batch, time() - t2, loss.item(), total_loss / it)
                )

        logging.info(
            'Train Epoch %04d | Time %.1fs | Loss %.4f'
            % (epoch, time() - t1, total_loss)
        )
        epoch_times.append(time() - t1)

        if epoch % args.evaluate_every == 0 or epoch == args.n_epoch:
            t3 = time()
            _, metrics = evaluate(
                model, data, Ks, device,
                args.eval_item_batch_size,
                compute_auc=(args.eval_skip_auc == 0),
            )
            eval_elapsed = time() - t3
            eval_times.append(eval_elapsed)
            auc_v = metrics[Ks[0]]['AUC']
            auc_log = (
                'N/A' if (isinstance(auc_v, float) and np.isnan(auc_v))
                else _fmt_metric_over_k(metrics, Ks, 'AUC')
            )
            logging.info(
                'Eval Epoch %04d | Time %.1fs | Ks=%s | '
                'Precision %s  Recall %s  NDCG %s  Hit %s  MRR %s  AUC %s'
                % (epoch, eval_elapsed, str(Ks),
                   _fmt_metric_over_k(metrics, Ks, 'precision'),
                   _fmt_metric_over_k(metrics, Ks, 'recall'),
                   _fmt_metric_over_k(metrics, Ks, 'ndcg'),
                   _fmt_metric_over_k(metrics, Ks, 'hit rate'),
                   _fmt_metric_over_k(metrics, Ks, 'mrr'),
                   auc_log)
            )

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg', 'hit rate', 'mrr', 'AUC']:
                    metrics_list[k][m].append(metrics[k][m])

            best_recall, should_stop = early_stopping(
                metrics_list[k_min]['recall'], args.stopping_steps
            )
            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Saved model at epoch %d' % epoch)
                best_epoch = epoch
                conv_time  = time()

    # Save metrics CSV
    rows = [epoch_list]
    cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg', 'hit rate', 'mrr', 'AUC']:
            rows.append(metrics_list[k][m])
            cols.append('{}@{}'.format(m.replace(' ', '_'), k))
    df = pd.DataFrame(rows).transpose()
    df.columns = cols
    df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    if best_epoch >= 0 and best_epoch in df['epoch_idx'].values:
        best = df.loc[df['epoch_idx'] == best_epoch].iloc[0].to_dict()
        def _col(m, kk):
            return '{}@{}'.format(m.replace(' ', '_'), kk)
        def _best_series(mkey):
            return '[' + ', '.join(
                ('nan' if (isinstance(best[_col(mkey, kk)], float) and np.isnan(best[_col(mkey, kk)]))
                 else '%.4f' % float(best[_col(mkey, kk)]))
                for kk in Ks
            ) + ']'

        b_auc0 = best[_col('AUC', Ks[0])]
        auc_best = 'N/A' if (isinstance(b_auc0, float) and np.isnan(b_auc0)) else _best_series('AUC')
        logging.info(
            'Best Eval Epoch %04d | Ks=%s | Precision %s  Recall %s  NDCG %s  Hit %s  MRR %s  AUC %s'
            % (int(best['epoch_idx']), str(Ks),
               _best_series('precision'), _best_series('recall'), _best_series('ndcg'),
               _best_series('hit rate'), _best_series('mrr'), auc_best)
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


if __name__ == '__main__':
    args = parse_ngcf_args()
    train(args)
