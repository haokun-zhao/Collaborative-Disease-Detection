"""Ranking metrics for MIMIC-IV baseline (aligned with CDD utility/metrics + batch_test)."""

import heapq
import numpy as np
from sklearn.metrics import roc_auc_score


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return float(np.mean(r))


def recall_at_k(r, k, all_pos_num):
    if all_pos_num <= 0:
        return 0.0
    r = np.asarray(r)[:k]
    return float(np.sum(r)) / float(all_pos_num)


def dcg_at_k(r, k, method=1):
    r = np.asarray(r)[:k]
    if r.size == 0:
        return 0.0
    if method == 1:
        return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))
    return float(r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1))))


def ndcg_at_k(r, k, ground_truth, method=1):
    gt = set(ground_truth)
    if len(gt) == 0:
        return 0.0
    if len(gt) > k:
        sent_list = [1.0] * k
    else:
        sent_list = [1.0] * len(gt) + [0.0] * (k - len(gt))
    dcg_max = dcg_at_k(sent_list, k, method)
    if dcg_max == 0:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def hit_at_k(r, k):
    r = np.array(r)[:k]
    return 1.0 if np.sum(r) > 0 else 0.0


def mrr_at_k(r, k):
    """
    Mean Reciprocal Rank at K.
    Returns 1/rank of the first hit in the top-K list, or 0 if no hit.
    Rank is 1-indexed.
    """
    r = np.asarray(r)[:k]
    hit_positions = np.where(r > 0)[0]
    if len(hit_positions) == 0:
        return 0.0
    return 1.0 / float(hit_positions[0] + 1)


def auc_ranking(user_pos_test, item_score_dict):
    """item_score_dict: item_id -> score (higher is better). Same construction as CDD batch_test.get_auc."""
    item_score = sorted(item_score_dict.items(), key=lambda kv: kv[1], reverse=True)
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]
    r = [1 if i in user_pos_test else 0 for i in item_sort]
    if len(set(r)) < 2:
        # No positive candidates in the pool → treat as random (0.5), not failure (0.0)
        return 0.5
    try:
        return float(roc_auc_score(y_true=r, y_score=posterior))
    except Exception:
        return 0.5


def rank_and_metrics(user_pos_test, test_items, rating_vec, ks):
    """
    :param user_pos_test: set or list of ground-truth positive item ids (test)
    :param test_items: candidate item ids (e.g. all items minus train)
    :param rating_vec: np array shape (n_items,) scores per item
    :param ks: list of K for top-K metrics
    :return: dict with auc, precision, recall, ndcg, hit_ratio, mrr (each a list over ks)
    """
    item_score = {i: float(rating_vec[i]) for i in test_items}
    k_max = max(ks)
    top_items = heapq.nlargest(k_max, item_score, key=item_score.get)
    r = [1 if i in user_pos_test else 0 for i in top_items]
    auc = auc_ranking(set(user_pos_test), item_score)

    out = {"auc": auc, "precision": [], "recall": [], "ndcg": [], "hit_ratio": [], "mrr": []}
    for k in ks:
        out["precision"].append(precision_at_k(r, k))
        out["recall"].append(recall_at_k(r, k, len(set(user_pos_test))))
        out["ndcg"].append(ndcg_at_k(r, k, user_pos_test))
        out["hit_ratio"].append(hit_at_k(r, k))
        out["mrr"].append(mrr_at_k(r, k))
    return out
