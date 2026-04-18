'''
Created on March 24, 2023

@author: Haokun Zhao (18392891510@163.com)
'''

import os
import random
import tracemalloc
import torch
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
import pandas as pd

from CDD import CDD
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.pyplot as plt


def _fmt_metric_list(x):
    """Format a scalar or per-K metric as a bracketed list for logging."""
    return '[' + ', '.join('%.5f' % float(v) for v in np.atleast_1d(x).ravel()) + ']'


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    feature_matrix = sp.load_npz(
        os.path.join(args.data_path, args.dataset, 'feature.npz')
    )

    args.node_dropout = eval(args.node_dropout) # [0.1]
    args.mess_dropout = eval(args.mess_dropout) # [0.1,0.1,0.1]
    print(args.device)  # cuda or cpu

    model = CDD(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 feature_matrix,
                 args)
    model.to(args.device)
    print("start to train")
    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, mrr_loger, auc_loger = [], [], [], [], [], [], []
    ls = []
    
    # load checkpoint if any
    checkpoint_path = os.path.join(args.weights_path, '259.pkl')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # efficiency tracking
    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)
    train_start  = time()
    epoch_times  = []
    eval_times   = []
    conv_time    = None
    best_epoch   = -1
    n_test_users = len(data_generator.test_set)

    # training process
    for epoch in range(start_epoch, args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        # print(n_batch) # 1026

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()   # 3 lists
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)    # forwarding

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # loss = mf_loss + emb_loss
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        # record the loss for each epoch
        ls.append(loss)
        epoch_times.append(time() - t1)

        if epoch == 0:
            print("Epoch 0: loss: %.5f" % loss)
        if (epoch + 1) % 10 != 0 and epoch != 0: # start from 0
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: training loss==[%.5f=%.5f + %.5f]' % (
                    epoch+1, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()

        # testing process
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

        if args.verbose > 0:
            perf_str = (
                'Epoch %d [%.1fs + %.1fs]: training loss=[%.5f=%.5f + %.5f], recall=%s, '
                'precision=%s, hit=%s, ndcg=%s, mrr=%s, auc=%s'
                % (epoch + 1, t2 - t1, t3 - t2, loss, mf_loss, emb_loss,
                   _fmt_metric_list(ret['recall']), _fmt_metric_list(ret['precision']),
                   _fmt_metric_list(ret['hit_ratio']), _fmt_metric_list(ret['ndcg']),
                   _fmt_metric_list(ret['mrr']), _fmt_metric_list(ret['auc'])))
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')
            best_epoch = epoch
            conv_time  = time()

    # ── efficiency summary ────────────────────────────────────────────────────
    peak_ram_mb  = tracemalloc.get_traced_memory()[1] / 1024 ** 2
    tracemalloc.stop()
    peak_gpu_mb  = (torch.cuda.max_memory_allocated(args.device) / 1024 ** 2
                    if torch.cuda.is_available() else 0.0)
    avg_epoch_s    = float(np.mean(epoch_times))    if epoch_times else 0.0
    avg_throughput = (float(n_test_users) / float(np.mean(eval_times))
                      if eval_times else 0.0)
    time_to_conv   = (conv_time - train_start) if conv_time is not None else (time() - train_start)

    print('=== Efficiency Summary ===')
    print('  Avg epoch time        : {:.2f} s'.format(avg_epoch_s))
    print('  Time to convergence   : {:.1f} s  (best epoch {:d})'.format(time_to_conv, best_epoch))
    print('  Peak RAM              : {:.1f} MB'.format(peak_ram_mb))
    print('  Peak GPU memory       : {:.1f} MB'.format(peak_gpu_mb))
    print('  Inference throughput  : {:.1f} users/s'.format(avg_throughput))

    os.makedirs(args.weights_path, exist_ok=True)
    eff_df = pd.DataFrame([{
        'avg_epoch_time_s':             avg_epoch_s,
        'time_to_convergence_s':        time_to_conv,
        'best_epoch':                   best_epoch,
        'peak_ram_mb':                  peak_ram_mb,
        'peak_gpu_mb':                  peak_gpu_mb,
        'inference_throughput_users_s': avg_throughput,
    }])
    eff_df.to_csv(os.path.join(args.weights_path, 'efficiency.tsv'), sep='\t', index=False)

    if rec_loger:
        recs  = np.array(rec_loger)
        pres  = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        hit   = np.array(hit_loger)
        mrrs  = np.array(mrr_loger)
        auc   = np.array(auc_loger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        def _fmt_metric_row(arr, i):
            # ret['auc'] is a scalar per epoch; recall/precision/etc. are per-K vectors
            return '\t'.join(['%.5f' % float(x) for x in np.atleast_1d(arr[i]).ravel()])

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s], mrr=[%s], auc=[%s]" % \
                     (idx, time() - t0, _fmt_metric_row(recs, idx), _fmt_metric_row(pres, idx),
                      _fmt_metric_row(hit, idx), _fmt_metric_row(ndcgs, idx), _fmt_metric_row(mrrs, idx),
                      _fmt_metric_row(auc, idx))
        print(final_perf)

    # Generate detailed report with individual patient results
    
    if os.environ.get("CDD_SKIP_DETAILED_REPORT", "0") != "1":
        print("\nGenerating detailed patient results report...")
        users_to_test = list(data_generator.test_set.keys())
        ret, detailed_results = test(model, users_to_test, drop_flag=False, batch_test_flag=True, return_detailed=True)
        generate_detailed_report(detailed_results,
                                 output_file='disease_prediction_result4.txt',
                                 csv_file='high_accuracy_patients4.csv')

    # Plot the loss
    # losses_np = [loss.detach().cpu().numpy() for loss in ls]
    # plt.rcParams['figure.figsize'] = (7.5, 2.5)
    # plt.plot(np.linspace(0, args.epoch, len(losses_np)), losses_np)
    # plt.xlabel('Epochs')
    # plt.ylabel('Training Loss')
    # plt.savefig('loss.png')
    # plt.show()
