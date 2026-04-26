"""
LightGCN training script.
Supports iterating over multiple seeds automatically (--run_all_seeds 1)
or running a single seed (--seed N).

Usage examples
--------------
# single run
python main.py --dataset mimicIV --seed 42

# all three seeds in one call
python main.py --dataset mimicIV --run_all_seeds 1

# eICU, all seeds
python main.py --dataset eICU --run_all_seeds 1
"""

import os
import sys
import copy
import random
import tracemalloc
from time import time

import numpy as np
import torch
import torch.optim as optim
import scipy.sparse as sp

# ── local imports ─────────────────────────────────────────────────────────────
from LightGCN import LightGCN
from utility.helper import early_stopping
from utility.batch_test import test, data_generator, args, Ks


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _fmt(x):
    return '[' + ', '.join('%.5f' % float(v) for v in np.atleast_1d(x).ravel()) + ']'


def _row(arr, i):
    return '\t'.join('%.5f' % float(x) for x in np.atleast_1d(arr[i]).ravel())


# ── single-seed training ──────────────────────────────────────────────────────

def run_one_seed(seed: int, base_args):
    """Train LightGCN with a given seed; return the best result dict."""
    set_seed(seed)

    # Build weight path that encodes dataset + seed so runs don't clobber each other
    weights_dir = os.path.join(
        base_args.weights_path,
        base_args.dataset,
        f'seed{seed}'
    )
    os.makedirs(weights_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  LightGCN | dataset={base_args.dataset} | seed={seed}')
    print(f'{"="*60}')

    device = base_args.device

    # adjacency matrix ─────────────────────────────────────────────────────────
    _, norm_adj, _ = data_generator.get_adj_mat()

    # model ────────────────────────────────────────────────────────────────────
    model = LightGCN(
        data_generator.n_users,
        data_generator.n_items,
        norm_adj,
        base_args,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=base_args.lr)

    # logging
    loss_loger, pre_loger, rec_loger, ndcg_loger = [], [], [], []
    hit_loger, mrr_loger, auc_loger = [], [], []

    cur_best_pre_0, stopping_step = 0., 0
    best_result   = None
    best_epoch    = -1
    conv_time     = None
    epoch_times, eval_times = [], []

    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time()

    # ── training loop ──────────────────────────────────────────────────────────
    for epoch in range(base_args.epoch):
        t1 = time()
        loss = mf_loss = emb_loss = 0.
        n_batch = data_generator.n_train // base_args.batch_size + 1

        for _ in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_emb, pos_emb, neg_emb = model(
                users, pos_items, neg_items, drop_flag=False
            )
            batch_loss, batch_mf, batch_emb = model.create_bpr_loss(
                u_emb, pos_emb, neg_emb
            )
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss     += batch_loss.item()
            mf_loss  += batch_mf.item()
            emb_loss += batch_emb.item()

        epoch_times.append(time() - t1)

        # evaluate every 10 epochs
        if (epoch + 1) % 10 != 0 and epoch != 0:
            if base_args.verbose > 0 and epoch % base_args.verbose == 0:
                print('Epoch %d [%.1fs]: loss=[%.5f=%.5f+%.5f]' % (
                    epoch + 1, time() - t1, loss, mf_loss, emb_loss))
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)
        t3 = time()
        eval_times.append(t3 - t2)

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        mrr_loger.append(ret['mrr'])
        auc_loger.append(ret['auc'])

        if base_args.verbose > 0:
            print(
                'Epoch %d [%.1fs+%.1fs]: loss=[%.5f=%.5f+%.5f], '
                'recall=%s, precision=%s, hit=%s, ndcg=%s, mrr=%s, auc=%.5f'
                % (epoch + 1, t2 - t1, t3 - t2, loss, mf_loss, emb_loss,
                   _fmt(ret['recall']), _fmt(ret['precision']),
                   _fmt(ret['hit_ratio']), _fmt(ret['ndcg']),
                   _fmt(ret['mrr']), ret['auc'])
            )

        cur_best_pre_0, stopping_step, should_stop = early_stopping(
            ret['recall'][0], cur_best_pre_0, stopping_step,
            expected_order='acc', flag_step=5
        )

        if ret['recall'][0] == cur_best_pre_0 and base_args.save_flag:
            ckpt_path = os.path.join(weights_dir, f'{epoch}.pkl')
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, ckpt_path)
            best_epoch = epoch
            conv_time  = time()
            best_result = {k: v.copy() if isinstance(v, np.ndarray) else v
                           for k, v in ret.items()}

        if should_stop:
            break

    # ── efficiency summary ─────────────────────────────────────────────────────
    peak_ram_mb = tracemalloc.get_traced_memory()[1] / 1024 ** 2
    tracemalloc.stop()
    peak_gpu_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                   if torch.cuda.is_available() else 0.)
    avg_epoch_s    = float(np.mean(epoch_times)) if epoch_times else 0.
    avg_throughput = (len(data_generator.test_set) / float(np.mean(eval_times))
                      if eval_times else 0.)
    ttc = (conv_time - train_start) if conv_time else (time() - train_start)

    print('\n=== Efficiency Summary (seed=%d) ===' % seed)
    print('  Avg epoch time      : {:.2f} s'.format(avg_epoch_s))
    print('  Time to convergence : {:.1f} s  (best epoch {:d})'.format(ttc, best_epoch))
    print('  Peak RAM            : {:.1f} MB'.format(peak_ram_mb))
    print('  Peak GPU memory     : {:.1f} MB'.format(peak_gpu_mb))
    print('  Inference throughput: {:.1f} users/s'.format(avg_throughput))

    # ── best-epoch result ──────────────────────────────────────────────────────
    if rec_loger:
        recs  = np.array(rec_loger)
        pres  = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        hits  = np.array(hit_loger)
        mrrs  = np.array(mrr_loger)
        aucs  = np.array(auc_loger)

        best_idx = int(np.argmax(recs[:, 0]))
        final_perf = (
            'Best Iter=[%d]@[%.1fs]\t'
            'recall=[%s]\tprecision=[%s]\thit=[%s]\tndcg=[%s]\tmrr=[%s]\tauc=%.5f'
            % (best_idx, time() - train_start,
               _row(recs, best_idx), _row(pres, best_idx),
               _row(hits, best_idx), _row(ndcgs, best_idx),
               _row(mrrs, best_idx), aucs[best_idx])
        )
        print(final_perf)

        # Save to results file
        result_file = os.path.join(
            weights_dir, f'results_seed{seed}.txt'
        )
        with open(result_file, 'w') as f:
            f.write(f'LightGCN | dataset={base_args.dataset} | seed={seed}\n')
            f.write(f'Ks={Ks}\n\n')
            for k_idx, K in enumerate(Ks):
                f.write(f'@{K:>2}: '
                        f'AUC={aucs[best_idx]:.5f}  '
                        f'Recall={recs[best_idx, k_idx]:.5f}  '
                        f'Precision={pres[best_idx, k_idx]:.5f}  '
                        f'NDCG={ndcgs[best_idx, k_idx]:.5f}  '
                        f'HitRate={hits[best_idx, k_idx]:.5f}  '
                        f'MRR={mrrs[best_idx, k_idx]:.5f}\n')
        print(f'Results written to {result_file}')

        return {
            'seed':      seed,
            'best_idx':  best_idx,
            'recall':    recs[best_idx],
            'precision': pres[best_idx],
            'ndcg':      ndcgs[best_idx],
            'hit_ratio': hits[best_idx],
            'mrr':       mrrs[best_idx],
            'auc':       float(aucs[best_idx]),
        }

    return None


# ── aggregation over seeds ────────────────────────────────────────────────────

def summarise_seeds(all_results, dataset, weights_path):
    """Print mean ± std across seeds and write a summary TSV."""
    if not all_results:
        return

    metrics_keys = ['recall', 'precision', 'ndcg', 'hit_ratio', 'mrr']
    seeds  = [r['seed'] for r in all_results]
    n_Ks   = len(all_results[0]['recall'])

    lines  = [f'\n{"="*70}']
    lines += [f'  LightGCN SUMMARY | dataset={dataset} | seeds={seeds}']
    lines += [f'{"="*70}']
    header = f"{'Metric':<15}" + ''.join(f'  @{Ks[i]:>2}  ' for i in range(n_Ks))
    lines.append(header)

    for mk in metrics_keys:
        vals = np.array([r[mk] for r in all_results])   # shape (n_seeds, n_Ks)
        row  = f'{mk:<15}'
        for i in range(n_Ks):
            row += '  {:.4f}±{:.4f}'.format(vals[:, i].mean(), vals[:, i].std())
        lines.append(row)

    # AUC is scalar
    aucs = np.array([r['auc'] for r in all_results])
    lines.append(f"{'auc':<15}  {aucs.mean():.4f}±{aucs.std():.4f}")

    summary = '\n'.join(lines)
    print(summary)

    # TSV
    summary_path = os.path.join(weights_path, dataset, 'summary.tsv')
    with open(summary_path, 'w') as f:
        f.write(f'metric\t' + '\t'.join([f'@{K}' for K in Ks]) + '\n')
        for mk in metrics_keys:
            vals = np.array([r[mk] for r in all_results])
            row  = mk + '\t' + '\t'.join(
                '{:.5f}±{:.5f}'.format(vals[:, i].mean(), vals[:, i].std())
                for i in range(n_Ks)
            )
            f.write(row + '\n')
        f.write('auc\t' + '\t'.join(
            ['{:.5f}±{:.5f}'.format(aucs.mean(), aucs.std())] * len(Ks)
        ) + '\n')
    print(f'Summary written to {summary_path}')


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args.device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {args.device}')

    if args.run_all_seeds:
        seed_list = eval(args.seeds)
    else:
        seed_list = [args.seed]

    all_results = []
    for s in seed_list:
        result = run_one_seed(s, args)
        if result is not None:
            all_results.append(result)

    if len(all_results) > 1:
        summarise_seeds(all_results, args.dataset, args.weights_path)
