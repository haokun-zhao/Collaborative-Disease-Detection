'''
Created on March 24, 2023

@author: Haokun Zhao (18392891510@163.com)
'''

import torch
import torch.optim as optim

from CDD import CDD
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.pyplot as plt


if __name__ == '__main__':

    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    feature_matrix = sp.load_npz('../Data/mimicIV/feature.npz')

    # shortest_path_file = sp.load_npz('../Data/mimicIV/shortest_path_grapg.npz')

    args.node_dropout = eval(args.node_dropout) # [0.1]
    args.mess_dropout = eval(args.mess_dropout) # [0.1,0.1,0.1]
    print(args.device)  # cuda:0

    model = CDD(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                #  shortest_path_file,
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

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, auc_loger, acc_loger = [], [], [], [], [], [], []
    ls = []

    # if os.path.exists(args.weights_path + str(0) + '.pkl'):
    #     checkpoint = torch.load(args.weights_path + str(0) + '.pkl')
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch']
    # else:
    start_epoch = 0

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
        users_to_test = list(data_generator.test_set.keys())    # [user id in the test set] [0,1,2,...,61254]
        # print("users_to_test: ", users_to_test)
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        auc_loger.append(ret['auc'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: training loss=[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f], auc=[%.5f]' % \
                       (epoch+1, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
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
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl') # save model parameters
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')
    
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    auc = np.array(auc_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s], auc=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
                  '\t'.join(['%.5f' % r for r in auc[idx]]))
    print(final_perf)

    # Plot the loss
    losses_np = [loss.detach().cpu().numpy() for loss in ls]
    plt.rcParams['figure.figsize'] = (7.5, 2.5)
    plt.plot(np.linspace(0, args.epoch, len(losses_np)), losses_np)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.savefig('loss.png')
    plt.show()
