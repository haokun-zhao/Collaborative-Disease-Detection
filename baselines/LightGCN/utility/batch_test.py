"""
Batch evaluation for LightGCN.
Reports 6 metrics at each K in Ks: AUC, Recall, Precision, NDCG, Hit Rate, MRR.
"""

import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import Data
import multiprocessing
import heapq
import numpy as np
import torch

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks   = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST   = data_generator.n_train, data_generator.n_test
BATCH_SIZE        = args.batch_size


# ───────────────────────────────────────────────────── ranking helpers ──────

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1], reverse=True)
    item_sort  = [x[0] for x in item_score]
    posterior  = [x[1] for x in item_score]
    r = [1 if i in user_pos_test else 0 for i in item_sort]
    return metrics.AUC(ground_truth=r, prediction=posterior)


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score  = {i: rating[i] for i in test_items}
    K_max       = max(Ks)
    top_items   = heapq.nlargest(K_max, item_score, key=item_score.get)
    r           = [1 if i in user_pos_test else 0 for i in top_items]
    auc         = get_auc(item_score, user_pos_test)
    return r, auc, top_items


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio, mrr = [], [], [], [], []
    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(metrics.hit_at_k(r, K))
        mrr.append(metrics.mrr_at_k(r, K))
    return {
        'recall':    np.array(recall),
        'precision': np.array(precision),
        'ndcg':      np.array(ndcg),
        'hit_ratio': np.array(hit_ratio),
        'mrr':       np.array(mrr),
        'auc':       auc,
    }


def test_one_user(x):
    rating, u = x[0], x[1]
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    user_pos_test = data_generator.test_set[u]
    all_items     = set(range(ITEM_NUM))
    test_items    = list(all_items - set(training_items))
    r, auc, top_items = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    perf = get_performance(user_pos_test, r, auc, Ks)
    perf["user_id"] = int(u)
    perf["test_items_count"] = int(len(user_pos_test))
    # Extra fields for downstream diagnostic analyses (head/mid/tail, clinical groups, density).
    # Keep lightweight: just ids and the top-K list already computed.
    perf["training_items"] = training_items
    perf["test_items"] = user_pos_test
    perf["top_predicted"] = top_items
    return perf


# ─────────────────────────────────────────────────────── main test fn ───────

def test(model, users_to_test, drop_flag=False, batch_test_flag=False, return_detailed=False):
    result = {
        'precision': np.zeros(len(Ks)),
        'recall':    np.zeros(len(Ks)),
        'ndcg':      np.zeros(len(Ks)),
        'hit_ratio': np.zeros(len(Ks)),
        'mrr':       np.zeros(len(Ks)),
        'auc':       0.,
    }
    pool = multiprocessing.Pool(cores)
    detailed = []

    u_batch_size = BATCH_SIZE * 2
    n_test_users  = len(users_to_test)
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0

    for u_batch_id in range(n_user_batchs):
        start      = u_batch_id * u_batch_size
        end        = (u_batch_id + 1) * u_batch_size
        user_batch = users_to_test[start:end]
        if not user_batch:
            continue

        if batch_test_flag:
            i_batch_size = BATCH_SIZE
            rate_batch   = np.zeros((len(user_batch), ITEM_NUM))
            i_count      = 0
            for i_batch_id in range(ITEM_NUM // i_batch_size + 1):
                i_start = i_batch_id * i_batch_size
                i_end   = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)
                item_batch = list(range(i_start, i_end))
                u_emb, pos_emb, _ = model(user_batch, item_batch, [], drop_flag=False)
                i_rate = model.rating(u_emb, pos_emb).detach().cpu()
                rate_batch[:, i_start:i_end] = i_rate.numpy()
                i_count += i_rate.shape[1]
                del u_emb, pos_emb, i_rate
                torch.cuda.empty_cache()
            assert i_count == ITEM_NUM
        else:
            item_batch = list(range(ITEM_NUM))
            u_emb, pos_emb, _ = model(user_batch, item_batch, [], drop_flag=False)
            rate_batch = model.rating(u_emb, pos_emb).detach().cpu()
            del u_emb, pos_emb
            torch.cuda.empty_cache()

        rate_np = rate_batch.numpy() if isinstance(rate_batch, torch.Tensor) else rate_batch
        batch_result = pool.map(test_one_user, zip(rate_np, user_batch))
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall']    += re['recall']    / n_test_users
            result['ndcg']      += re['ndcg']      / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['mrr']       += re['mrr']       / n_test_users
            result['auc']       += re['auc']       / n_test_users
            if return_detailed:
                detailed.append(re)

    assert count == n_test_users
    pool.close()
    if return_detailed:
        return result, detailed
    return result
