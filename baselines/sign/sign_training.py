# Copyright 2020 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
SIGN training script.

Supports two modes:
  - Original OGB (ogbn-papers100M) mode  : python sign_training.py
  - MIMIC-IV recommendation mode         : python sign_training.py --mimic

For MIMIC, run mimic_preprocessing.py first to generate the embeddings file:
    python mimic_preprocessing.py            # produces mimic_sign.pt
    python sign_training.py --mimic          # trains and evaluates
"""

import argparse
import heapq

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from logger import Logger

# OGB is only needed for the original (non-MIMIC) mode
try:
    from ogb.nodeproppred import Evaluator as OGBEvaluator
    _has_ogb = True
except ImportError:
    _has_ogb = False


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert self.x.size(0) == self.y.size(0)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class UidFeatureDataset(Dataset):
    """Lazy-index dataset: stores a reference to the full feature tensor and
    an index array; materialises nothing upfront.  Returns (uid, feature_row).
    Works with float16 feature tensors – callers cast to float32 per-batch.
    """
    def __init__(self, x, idx):
        self.x   = x    # (n_users, feat_dim) – may be float16
        self.idx = idx  # 1-D LongTensor of user indices for this split

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        uid = int(self.idx[i])
        return uid, self.x[uid]   # single-row view, no bulk copy


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_sigmoid=False):
        super(MLP, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        if self.use_sigmoid:
            return x  # raw logits; BCE loss is applied outside
        return torch.log_softmax(x, dim=-1)


# ---------------------------------------------------------------------------
# Original OGB training / evaluation
# ---------------------------------------------------------------------------

def train(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, device, loader, evaluator):
    model.eval()
    y_pred, y_true = [], []
    for x, y in tqdm(loader):
        x = x.to(device)
        out = model(x)
        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)
    return evaluator.eval({
        "y_true": torch.cat(y_true, dim=0),
        "y_pred": torch.cat(y_pred, dim=0),
    })['acc']


# ---------------------------------------------------------------------------
# MIMIC-IV training / evaluation
# ---------------------------------------------------------------------------

def train_mimic(model, device, train_loader, optimizer, train_items, n_items):
    """One epoch of BPR training using a UidFeatureDataset loader.

    train_loader yields (uids, x_batch) where x_batch may be float16.
    Positives are looked up from train_items dict – no dense label tensor needed.
    """
    model.train()
    total_loss, n = 0.0, 0
    for uids, x_batch in train_loader:
        x_batch = x_batch.to(device).float()           # fp16 → fp32 per batch
        optimizer.zero_grad()
        out = model(x_batch)                            # (batch, n_items) raw logits

        # Build BPR pairs from train_items dict (avoids 490 MB dense label tensor)
        pos_row, pos_col = [], []
        for local_i, uid in enumerate(uids.tolist()):
            for iid in train_items.get(uid, []):
                pos_row.append(local_i)
                pos_col.append(iid)
        if not pos_row:
            continue

        pos_u = torch.tensor(pos_row, device=device)
        pos_i = torch.tensor(pos_col, device=device)
        neg_i = torch.randint(0, n_items, (len(pos_i),), device=device)

        pos_sc = out[pos_u, pos_i]
        neg_sc = out[pos_u, neg_i]
        loss = F.softplus(neg_sc - pos_sc).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
        n += x_batch.size(0)
    return total_loss / n if n > 0 else 0.0


@torch.no_grad()
def evaluate_mimic(model, device, x_all, train_items, test_set, n_items, ks,
                   eval_batch_size=512):
    """
    Compute ranking metrics on the test set.
    Evaluation protocol (mirrors CDD): for each test user, candidates = all items
    minus train items; rank by predicted score; compute AUC, Precision, Recall,
    NDCG, Hit Rate at each K in ks.
    """
    model.eval()
    n_users = x_all.shape[0]

    # Batch inference to avoid OOM.
    # x_all may be float16; cast per-batch to float32 for the model.
    # Logits are stored as float16 on CPU (~half the scratch memory of float32).
    import time as _time
    _t0 = _time.perf_counter()
    logits_list = []
    for start in range(0, n_users, eval_batch_size):
        x_batch = x_all[start:start + eval_batch_size].to(device).float()
        logits_list.append(model(x_batch).half().cpu())
    all_logits = torch.cat(logits_list, dim=0).float().numpy()   # (n_users, n_items)
    _inference_time = _time.perf_counter() - _t0

    sum_auc  = 0.0
    sum_prec = np.zeros(len(ks))
    sum_rec  = np.zeros(len(ks))
    sum_ndcg = np.zeros(len(ks))
    sum_hit  = np.zeros(len(ks))
    sum_mrr  = np.zeros(len(ks))
    n_ev = 0

    for uid, test_pos in test_set.items():
        train_pos = set(train_items.get(uid, []))
        candidates = [i for i in range(n_items) if i not in train_pos]
        if not candidates or not test_pos:
            continue

        scores = {i: float(all_logits[uid, i]) for i in candidates}
        sorted_items  = sorted(scores, key=scores.get, reverse=True)
        sorted_scores = [scores[i] for i in sorted_items]
        pos_set = set(test_pos)
        r_full = [1 if i in pos_set else 0 for i in sorted_items]

        try:
            auc = float(roc_auc_score(r_full, sorted_scores)) if len(set(r_full)) >= 2 else 0.5
        except Exception:
            auc = 0.5
        sum_auc += auc

        k_max = max(ks)
        r = r_full[:k_max]
        for j, k in enumerate(ks):
            rk = r[:k]
            sum_prec[j] += float(np.mean(rk))
            sum_rec[j]  += float(np.sum(rk)) / max(len(pos_set), 1)
            n_pos = len(pos_set)
            ideal = [1.0] * min(n_pos, k) + [0.0] * max(0, k - n_pos)
            ideal_dcg = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
            dcg       = sum(v / np.log2(i + 2) for i, v in enumerate(rk))
            sum_ndcg[j] += dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            sum_hit[j]  += 1.0 if sum(rk) > 0 else 0.0
            hit_pos = next((i for i, v in enumerate(rk) if v > 0), None)
            sum_mrr[j]  += 1.0 / (hit_pos + 1) if hit_pos is not None else 0.0
        n_ev += 1

    if n_ev == 0:
        return None
    return {
        "n_eval":     n_ev,
        "auc":        sum_auc  / n_ev,
        "precision":  (sum_prec / n_ev).tolist(),
        "recall":     (sum_rec  / n_ev).tolist(),
        "ndcg":       (sum_ndcg / n_ev).tolist(),
        "hit":        (sum_hit  / n_ev).tolist(),
        "mrr":                (sum_mrr  / n_ev).tolist(),
        "ks":                 ks,
        "inference_time_s":   round(_inference_time, 3),
        "inference_throughput": round(n_users / _inference_time, 1) if _inference_time > 0 else 0,
    }


def main_mimic(args, device):
    """MIMIC-IV training and evaluation loop.

    Memory-efficient design:
      - op_embedding stored as float16; cast to float32 per batch.
      - No dense label tensor; positives looked up from train_items dict.
      - Lazy UidFeatureDataset: no upfront copy of train/val subsets.
      - op_dict freed immediately after extracting tensors.
    """
    try:
        op_dict = torch.load(args.embeddings_file_name, weights_only=False)
    except TypeError:
        op_dict = torch.load(args.embeddings_file_name)

    if "n_items" not in op_dict:
        raise ValueError(
            f"'{args.embeddings_file_name}' does not look like a MIMIC file. "
            "Run  python mimic_preprocessing.py  first."
        )

    # Concatenate propagated feature matrices along the feature dimension.
    # Individual hop tensors (float16) are freed right after concatenation.
    x = torch.cat(op_dict["op_embedding"], dim=1)   # (n_users, (K+1)*feat_dim) fp16
    n_users     = op_dict["n_users"]
    n_items     = op_dict["n_items"]
    train_items = op_dict["train_items"]
    test_set    = op_dict["test_set"]
    split_idx   = op_dict["split_idx"]
    del op_dict   # release the K+1 individual hop tensors (~2 GB for 3-hop MIMIC)

    ks = list(args.ks)

    print(f"MIMIC-IV SIGN")
    print(f"  n_users={n_users}  n_items={n_items}")
    print(f"  Input feature dimension : {x.shape[-1]}  dtype={x.dtype}")

    train_idx = split_idx["train"]
    val_idx   = split_idx["valid"]

    # Lazy datasets: hold a reference to x, no bulk copy of subsets
    train_dataset = UidFeatureDataset(x, train_idx)
    val_dataset   = UidFeatureDataset(x, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    model = MLP(
        in_channels=x.size(-1),
        hidden_channels=args.hidden_channels,
        out_channels=n_items,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_sigmoid=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params}")

    import time as _time
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_loss   = float("inf")
        convergence_epoch = args.epochs
        train_log       = []

        for epoch in range(1, 1 + args.epochs):
            _ep_t0 = _time.perf_counter()
            train_loss = train_mimic(model, device, train_loader, optimizer,
                                     train_items, n_items)

            # Validation BPR loss (consistent with training; no dense label tensor)
            model.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for uids_v, xb in val_loader:
                    xb = xb.to(device).float()             # fp16 → fp32
                    out = model(xb)
                    pos_row, pos_col = [], []
                    for local_i, uid in enumerate(uids_v.tolist()):
                        for iid in train_items.get(uid, []):
                            pos_row.append(local_i)
                            pos_col.append(iid)
                    if not pos_row:
                        continue
                    pos_u = torch.tensor(pos_row, device=device)
                    pos_i = torch.tensor(pos_col, device=device)
                    neg_i = torch.randint(0, n_items, (len(pos_i),), device=device)
                    vl = F.softplus(out[pos_u, neg_i] - out[pos_u, pos_i]).mean().item()
                    val_loss += vl * xb.size(0)
                    n_val    += xb.size(0)
            val_loss = val_loss / n_val if n_val > 0 else 0.0

            ep_time = _time.perf_counter() - _ep_t0
            train_log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                               "val_loss": round(val_loss, 6), "epoch_time_s": round(ep_time, 4)})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                convergence_epoch = epoch

            if epoch % args.log_steps == 0:
                print(f"Run {run+1:02d}  Epoch {epoch:03d}  "
                      f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  time={ep_time:.3f}s")

        # Final ranking evaluation on the test set
        metrics = evaluate_mimic(model, device, x, train_items, test_set, n_items, ks)
        if metrics is None:
            print(f"Run {run+1}: no test users evaluated.")
            continue
        print(f"\nRun {run+1} Test (evaluated users={metrics['n_eval']}):")
        print(f"  AUC: {metrics['auc']:.6f}  "
              f"  convergence_epoch={convergence_epoch}  "
              f"  throughput={metrics.get('inference_throughput', 0):.0f} users/s")
        for j, k in enumerate(ks):
            print(f"  K={k:2d}  Precision: {metrics['precision'][j]:.6f}"
                  f"  Recall: {metrics['recall'][j]:.6f}"
                  f"  NDCG: {metrics['ndcg'][j]:.6f}"
                  f"  Hit Rate: {metrics['hit'][j]:.6f}"
                  f"  MRR: {metrics['mrr'][j]:.6f}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# run sign
# python sign_training.py --device 0 --dropout 0.3 --lr 0.00005 --hidden_channels 512 --num_layers 3 --embeddings_file_name sign_333_embeddings.pt --result_file_name sign_results.txt

# run sign-xl
# python sign_training.py --device 1 --dropout 0.5 --lr 0.00005 --hidden_channels 2048 --num_layers 3 --embeddings_file_name sign_333_embeddings.pt --result_file_name sign-xl_results.txt

# run MIMIC
# python sign_training.py --mimic --embeddings_file_name mimic_sign.pt


def main():
    parser = argparse.ArgumentParser(description='SIGN')
    # Mode
    parser.add_argument('--mimic', action='store_true',
                        help='Run in MIMIC-IV recommendation mode (requires mimic_preprocessing.py output)')
    parser.add_argument('--ks', nargs='+', type=int, default=[3, 5, 10, 20],
                        help='MIMIC: top-K values for ranking metrics')
    # Common
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--embeddings_file_name', type=str, default='mimic_sign.pt')
    parser.add_argument('--result_file_name', type=str, default='results.txt')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.mimic:
        main_mimic(args, device)
        return

    # -----------------------------------------------------------------------
    # Original OGB mode
    # -----------------------------------------------------------------------
    if not _has_ogb:
        raise ImportError(
            "ogb package is required for OGB mode. "
            "Install with:  pip install ogb\n"
            "For MIMIC-IV, use the --mimic flag instead."
        )

    try:
        op_dict = torch.load(args.embeddings_file_name)
    except Exception:
        raise RuntimeError(
            f"File '{args.embeddings_file_name}' not found. "
            "Run  python preprocessing.py  first."
        )

    split_idx = op_dict['split_idx']
    x = torch.cat(op_dict['op_embedding'], dim=1)
    y = op_dict['label'].to(torch.long)
    num_classes = 172
    print('Input feature dimension: {}'.format(x.shape[-1]))
    print('Total number of nodes: {}'.format(x.shape[0]))

    train_dataset = SimpleDataset(x[split_idx['train']], y[split_idx['train']])
    valid_dataset = SimpleDataset(x[split_idx['valid']], y[split_idx['valid']])
    test_dataset  = SimpleDataset(x[split_idx['test']],  y[split_idx['test']])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

    model = MLP(x.size(-1), args.hidden_channels, num_classes,
                args.num_layers, args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {n_params}.')

    evaluator = OGBEvaluator(name='ogbn-papers100M')
    logger = Logger(args.runs, info=args, file_name=args.result_file_name)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            train(model, device, train_loader, optimizer)
            train_acc = test(model, device, train_loader, evaluator)
            valid_acc = test(model, device, valid_loader, evaluator)
            test_acc  = test(model, device, test_loader,  evaluator)
            logger.add_result(run, (train_acc, valid_acc, test_acc))
            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()
