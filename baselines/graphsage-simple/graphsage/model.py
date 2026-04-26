import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
import time
import random
import argparse
import heapq
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, roc_auc_score
import scipy.sparse as sp

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets, and MIMIC-IV disease recommendation.
"""


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


class MimicGraphSage(nn.Module):
    """Multi-label GraphSAGE head for disease recommendation (BCE loss)."""

    def __init__(self, num_items, enc):
        super(MimicGraphSage, self).__init__()
        self.enc = enc
        self.weight = nn.Parameter(torch.FloatTensor(num_items, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)          # (embed_dim, batch)
        scores = self.weight.mm(embeds)   # (num_items, batch)
        return scores.t()                 # (batch, num_items)

    def loss(self, nodes, labels):
        scores = self.forward(nodes)      # (batch, num_items)
        if not scores.isfinite().all():
            return scores.new_zeros(())   # guard: NaN/Inf in scores → zero loss
        # BPR loss: for each (user, pos_item) pair, sample one neg_item
        pos_u, pos_i = (labels > 0.5).nonzero(as_tuple=True)
        if pos_u.numel() == 0:
            return scores.new_zeros(())   # no positives in batch → zero loss
        neg_i = torch.randint(0, scores.size(1), (pos_u.numel(),))
        pos_sc = scores[pos_u, pos_i]
        neg_sc = scores[pos_u, neg_i]
        return nn.functional.softplus(neg_sc - pos_sc).mean()


# ---------------------------------------------------------------------------
# Cora
# ---------------------------------------------------------------------------

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if info[-1] not in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              torch.LongTensor(labels[np.array(batch_nodes)]))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

def load_pubmed():
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1]) - 1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              torch.LongTensor(labels[np.array(batch_nodes)]))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


# ---------------------------------------------------------------------------
# MIMIC-IV
# ---------------------------------------------------------------------------

def _load_interactions(path, store, n_users, n_items):
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            uid = int(parts[0])
            items = [int(float(x)) for x in parts[1:]]
            store[uid] = items
            n_users = max(n_users, uid)
            if items:
                n_items = max(n_items, max(items))
    return n_users, n_items


def load_mimic(data_dir):
    """
    Load MIMIC-IV data for GraphSAGE.
    Returns:
        feat_data   : np.ndarray (n_users+n_items, n_user_feats)   float32
        multi_hot   : np.ndarray (n_users, n_items)                float32
        adj_lists   : defaultdict(set)  node-id -> set of neighbor node-ids
        n_users     : int
        n_items     : int
        n_user_feats: int
        train_items : dict uid -> [item ids]
        test_set    : dict uid -> [item ids]
    """
    data_dir = Path(data_dir)
    train_items, test_set = {}, {}
    n_users, n_items = 0, 0
    n_users, n_items = _load_interactions(data_dir / "train2.txt", train_items, n_users, n_items)
    n_users, n_items = _load_interactions(data_dir / "test2.txt",  test_set,    n_users, n_items)
    n_users += 1
    n_items += 1

    # Bipartite adjacency lists (training edges only; items are offset by n_users)
    adj_lists = defaultdict(set)
    for uid, items in train_items.items():
        for iid in items:
            adj_lists[uid].add(n_users + iid)
            adj_lists[n_users + iid].add(uid)

    # Node features: user rows from feature.npz, item rows = zeros (identity captured via graph)
    feat_scipy = sp.load_npz(str(data_dir / "feature.npz")).astype(np.float32)
    n_user_feats = feat_scipy.shape[1]
    num_nodes = n_users + n_items
    feat_data = np.zeros((num_nodes, n_user_feats), dtype=np.float32)
    cx = feat_scipy.tocsr()
    for u in range(min(n_users, cx.shape[0])):
        row = cx[u]
        for j in row.indices:
            feat_data[u, j] = row.data[row.indices == j][0]

    # Multi-hot training labels for user nodes
    multi_hot = np.zeros((n_users, n_items), dtype=np.float32)
    for uid, items in train_items.items():
        for iid in items:
            multi_hot[uid, iid] = 1.0

    return feat_data, multi_hot, adj_lists, n_users, n_items, n_user_feats, train_items, test_set


def _rank_metrics(pos_set, candidates, scores_vec, ks):
    """Compute AUC, Precision, Recall, NDCG, Hit Rate, MRR at each K."""
    item_score = {i: float(scores_vec[i]) for i in candidates}
    sorted_items  = sorted(item_score, key=item_score.get, reverse=True)
    sorted_scores = [item_score[i] for i in sorted_items]
    r_full = [1 if i in pos_set else 0 for i in sorted_items]
    try:
        auc = float(roc_auc_score(r_full, sorted_scores)) if len(set(r_full)) >= 2 else 0.5
    except Exception:
        auc = 0.5

    k_max = max(ks)
    r = r_full[:k_max]
    result = {"auc": auc, "precision": [], "recall": [], "ndcg": [], "hit": [], "mrr": []}
    for k in ks:
        rk = r[:k]
        result["precision"].append(float(np.mean(rk)))
        result["recall"].append(float(np.sum(rk)) / max(len(pos_set), 1))
        n_pos    = len(pos_set)
        ideal    = [1.0] * min(n_pos, k) + [0.0] * max(0, k - n_pos)
        ideal_dcg = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
        dcg       = sum(v / np.log2(i + 2) for i, v in enumerate(rk))
        result["ndcg"].append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        result["hit"].append(1.0 if sum(rk) > 0 else 0.0)
        # MRR@K: reciprocal rank of the first hit within top-K
        hit_pos = next((i for i, v in enumerate(rk) if v > 0), None)
        result["mrr"].append(1.0 / (hit_pos + 1) if hit_pos is not None else 0.0)
    return result


def run_mimic(data_dir=None, epochs=50, batch_size=256, lr=0.01,
              embed_dim=128, num_samples=5, ks=(5, 20), seed=1):
    """
    Train and evaluate GraphSAGE on MIMIC-IV.
    Returns a dict with ranking metrics, timing, and epoch count for
    programmatic access (used by benchmark_mimic.py).
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "Data" / "mimicIV"

    print("Loading MIMIC-IV data ...")
    (feat_data, multi_hot, adj_lists,
     n_users, n_items, n_user_feats, train_items, test_set) = load_mimic(data_dir)
    num_nodes = n_users + n_items
    feat_dim = n_user_feats
    print(f"  n_users={n_users}  n_items={n_items}  feat_dim={feat_dim}  total_nodes={num_nodes}")

    features = nn.Embedding(num_nodes, feat_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, feat_dim, embed_dim, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, embed_dim, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = num_samples
    enc2.num_samples = num_samples

    graphsage = MimicGraphSage(n_items, enc2)

    # Only use users that have training interactions
    all_train_users = sorted(train_items.keys())
    random.shuffle(all_train_users)
    n_val = max(1, int(0.1 * len(all_train_users)))
    train_sup_users = all_train_users[n_val:]

    labels_tensor = torch.FloatTensor(multi_hot)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=lr
    )

    import time as _time
    train_log = []
    t_train_start = _time.perf_counter()
    for epoch in range(1, epochs + 1):
        _ep_t0 = _time.perf_counter()
        graphsage.train()
        batch = random.sample(train_sup_users, min(batch_size, len(train_sup_users)))
        optimizer.zero_grad()
        loss = graphsage.loss(batch, labels_tensor[batch])
        loss.backward()
        optimizer.step()
        ep_time = _time.perf_counter() - _ep_t0
        train_log.append({
            "epoch":        epoch,
            "train_loss":   round(float(loss.item()), 6),
            "epoch_time_s": round(ep_time, 4),
        })
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}  loss={loss.item():.4f}  time={ep_time:.3f}s")
    train_time_s = _time.perf_counter() - t_train_start

    # Evaluation
    print("\nEvaluating on test set ...")
    t_eval_start = _time.perf_counter()
    graphsage.eval()
    with torch.no_grad():
        chunk = 2048
        all_scores = []
        for start in range(0, n_users, chunk):
            end = min(start + chunk, n_users)
            all_scores.append(graphsage.forward(list(range(start, end))).numpy())
        all_scores = np.concatenate(all_scores, axis=0)   # (n_users, n_items)
    eval_time_s = _time.perf_counter() - t_eval_start

    ks_list = list(ks)
    sum_auc  = 0.0
    sum_prec = np.zeros(len(ks_list))
    sum_rec  = np.zeros(len(ks_list))
    sum_ndcg = np.zeros(len(ks_list))
    sum_hit  = np.zeros(len(ks_list))
    sum_mrr  = np.zeros(len(ks_list))
    n_ev = 0

    for uid, test_pos in test_set.items():
        train_pos = set(train_items.get(uid, []))
        candidates = [i for i in range(n_items) if i not in train_pos]
        if not candidates or not test_pos:
            continue
        m = _rank_metrics(set(test_pos), candidates, all_scores[uid], ks_list)
        sum_auc  += m["auc"]
        sum_prec += np.array(m["precision"])
        sum_rec  += np.array(m["recall"])
        sum_ndcg += np.array(m["ndcg"])
        sum_hit  += np.array(m["hit"])
        sum_mrr  += np.array(m["mrr"])
        n_ev += 1

    if n_ev == 0:
        print("No test users evaluated.")
        return {}

    print(f"\nMIMIC-IV test results  (evaluated users: {n_ev})")
    print(f"  AUC: {sum_auc/n_ev:.6f}")
    for j, k in enumerate(ks_list):
        print(f"  K={k:2d}  P={sum_prec[j]/n_ev:.6f}  R={sum_rec[j]/n_ev:.6f}"
              f"  NDCG={sum_ndcg[j]/n_ev:.6f}  Hit={sum_hit[j]/n_ev:.6f}  MRR={sum_mrr[j]/n_ev:.6f}")

    return {
        "n_eval":                  n_ev,
        "auc":                     float(sum_auc / n_ev),
        "precision":               [float(v / n_ev) for v in sum_prec],
        "recall":                  [float(v / n_ev) for v in sum_rec],
        "ndcg":                    [float(v / n_ev) for v in sum_ndcg],
        "hit":                     [float(v / n_ev) for v in sum_hit],
        "mrr":                     [float(v / n_ev) for v in sum_mrr],
        "ks":                      ks_list,
        "train_time_s":            train_time_s,
        "eval_time_s":             eval_time_s,
        "epochs":                  epochs,
        "time_per_epoch_s":        train_time_s / epochs,
        "inference_throughput":    round(n_users / eval_time_s, 1) if eval_time_s > 0 else 0,
        "train_log":               train_log,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mimic",
                        choices=["cora", "pubmed", "mimic"],
                        help="Dataset to run. Default: mimic")
    parser.add_argument("--data_dir", default=None,
                        help="Path to MIMIC-IV data directory (default: ../../Data/mimicIV)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of neighbors to sample per hop")
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10, 20],
                        help="Top-K values for ranking metrics")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    args = parser.parse_args()

    if args.dataset == "cora":
        run_cora()
    elif args.dataset == "pubmed":
        run_pubmed()
    else:
        run_mimic(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            embed_dim=args.embed_dim,
            num_samples=args.num_samples,
            ks=tuple(args.ks),
            seed=args.seed,
        )
