"""
Multi-seed MIMIC-IV benchmark for SIGN.

Metrics logged per seed:
  Ranking  : AUC, Precision, Recall, NDCG, Hit Rate, MRR  @  K ∈ {3,5,10,20}
  Timing   : total_time_s, train_time_s, eval_time_s,
             time_per_epoch_s (mean), inference_throughput (users/s)
  Convergence: convergence_epoch (epoch of best val-BCE), train_log saved to JSON
  Memory   : peak_ram_gb, peak_gpu_gb

Usage (from sign/):
    # Step 1 – precompute propagated features (only once)
    python mimic_preprocessing.py

    # Step 2 – run benchmark
    python benchmark_mimic.py                          # 5 default seeds
    python benchmark_mimic.py --seeds 42 123 456       # custom seeds
    python benchmark_mimic.py --hidden 512 --epochs 60

Output: per-seed results + mean ± std table + benchmark_results_sign.json

NOTE: Data split is fixed by mimic_preprocessing.py (seed=42). The 5 benchmark
seeds vary only the model initialisation and training stochasticity (dropout,
batch shuffling). This is consistent with how SIGN was originally evaluated.
"""

import sys
import os
import time
import json
import argparse
import gc

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

try:
    import psutil
    _has_psutil = True
except ImportError:
    _has_psutil = False

from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Local imports (run from sign/ directory)
from sign_training import MLP, SimpleDataset

DEFAULT_SEEDS  = [42, 123, 456, 789, 1234]
DEFAULT_KS     = [3, 5, 10, 20]
DEFAULT_PT     = "mimic_sign.pt"


# ── resource helpers ──────────────────────────────────────────────────────────

def _peak_ram_gb():
    if not _has_psutil:
        return float("nan")
    mem = psutil.Process().memory_info()
    return getattr(mem, "peak_wset", mem.rss) / 1024 ** 3


def _peak_gpu_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return 0.0


def _reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ── ranking metrics (same protocol as CDD) ────────────────────────────────────

def _evaluate_ranking(all_logits, train_items, test_set, n_items, ks):
    """all_logits: np.ndarray (n_users, n_items)"""
    n_users  = all_logits.shape[0]
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

        scores        = {i: float(all_logits[uid, i]) for i in candidates}
        sorted_items  = sorted(scores, key=scores.get, reverse=True)
        sorted_scores = [scores[i] for i in sorted_items]
        pos_set       = set(test_pos)
        r_full        = [1 if i in pos_set else 0 for i in sorted_items]

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
            n_pos   = len(pos_set)
            ideal   = [1.0] * min(n_pos, k) + [0.0] * max(0, k - n_pos)
            idcg    = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
            dcg     = sum(v / np.log2(i + 2) for i, v in enumerate(rk))
            sum_ndcg[j] += dcg / idcg if idcg > 0 else 0.0
            sum_hit[j]  += 1.0 if sum(rk) > 0 else 0.0
            hit_pos = next((i for i, v in enumerate(rk) if v > 0), None)
            sum_mrr[j]  += 1.0 / (hit_pos + 1) if hit_pos is not None else 0.0
        n_ev += 1

    if n_ev == 0:
        return None
    return {
        "n_eval":    n_ev,
        "auc":       float(sum_auc / n_ev),
        "precision": [float(v / n_ev) for v in sum_prec],
        "recall":    [float(v / n_ev) for v in sum_rec],
        "ndcg":      [float(v / n_ev) for v in sum_ndcg],
        "hit":       [float(v / n_ev) for v in sum_hit],
        "mrr":       [float(v / n_ev) for v in sum_mrr],
        "ks":        list(ks),
    }


# ── single-seed run ──────────────────────────────────────────────────────────

def run_one_seed(seed, shared, bargs, device):
    """
    Reinitialise model and train from scratch with the given seed.
    shared : dict with pre-loaded tensors and metadata.
    Returns flat dict with timing, memory, and ranking metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    _reset_gpu_stats()

    x           = shared["x"]           # (n_users, total_feat_dim)
    y           = shared["y"]           # (n_users, n_items) float32
    train_idx   = shared["train_idx"]
    val_idx     = shared["val_idx"]
    train_items = shared["train_items"]
    test_set    = shared["test_set"]
    n_users     = shared["n_users"]
    n_items     = shared["n_items"]
    ks          = bargs.ks

    train_ds = SimpleDataset(x[train_idx], y[train_idx])
    val_ds   = SimpleDataset(x[val_idx],   y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=bargs.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bargs.batch_size, shuffle=False)

    model = MLP(
        in_channels=x.size(-1),
        hidden_channels=bargs.hidden,
        out_channels=n_items,
        num_layers=bargs.num_layers,
        dropout=bargs.dropout,
        use_sigmoid=True,
    ).to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=bargs.lr)

    # ── training ─────────────────────────────────────────────────────────
    best_val_loss     = float("inf")
    convergence_epoch = bargs.epochs
    train_log         = []

    t_train_start = time.perf_counter()
    for epoch in range(1, bargs.epochs + 1):
        _ep_t0 = time.perf_counter()
        model.train()
        train_loss, n_tr = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n_tr += xb.size(0)
        train_loss /= max(n_tr, 1)

        # Validation loss
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += F.binary_cross_entropy_with_logits(model(xb), yb).item() * xb.size(0)
                n_val += xb.size(0)
        val_loss /= max(n_val, 1)
        ep_time = time.perf_counter() - _ep_t0

        train_log.append({
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_loss, 6),
            "epoch_time_s": round(ep_time, 4),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            convergence_epoch = epoch

        if epoch % 10 == 0:
            print(f"    epoch {epoch}/{bargs.epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  time={ep_time:.3f}s")

    train_time_s = time.perf_counter() - t_train_start
    time_per_epoch_s = train_time_s / bargs.epochs

    # ── evaluation ───────────────────────────────────────────────────────
    t_eval_start = time.perf_counter()
    model.eval()
    chunk = 512
    logits_list = []
    with torch.no_grad():
        for start in range(0, n_users, chunk):
            xb = x[start:start + chunk].to(device)
            logits_list.append(model(xb).cpu())
    all_logits = torch.cat(logits_list, dim=0).numpy()
    eval_time_s = time.perf_counter() - t_eval_start
    inference_throughput = round(n_users / eval_time_s, 1) if eval_time_s > 0 else 0

    metrics = _evaluate_ranking(all_logits, train_items, test_set, n_items, ks)
    del model, all_logits, logits_list
    gc.collect()

    if metrics is None:
        metrics = {"n_eval": 0, "auc": float("nan"),
                   "precision": [], "recall": [], "ndcg": [], "hit": [], "mrr": [], "ks": list(ks)}

    return {
        "seed":                  seed,
        "train_time_s":          round(train_time_s, 3),
        "eval_time_s":           round(eval_time_s,  3),
        "total_time_s":          round(train_time_s + eval_time_s, 3),
        "time_per_epoch_s":      round(time_per_epoch_s, 4),
        "epochs_run":            bargs.epochs,
        "convergence_epoch":     convergence_epoch,
        "inference_throughput":  inference_throughput,
        "peak_ram_gb":           round(_peak_ram_gb(), 3),
        "peak_gpu_gb":           round(_peak_gpu_gb(), 3),
        "train_log":             train_log,
        **metrics,
    }


# ── aggregation & reporting ───────────────────────────────────────────────────

def _scalar_fields(run):
    ks = run.get("ks", [])
    fields = {
        "total_time_s":         run["total_time_s"],
        "train_time_s":         run["train_time_s"],
        "eval_time_s":          run["eval_time_s"],
        "time_per_epoch_s":     run["time_per_epoch_s"],
        "epochs_run":           run["epochs_run"],
        "convergence_epoch":    run["convergence_epoch"],
        "inference_throughput": run["inference_throughput"],
        "peak_ram_gb":          run["peak_ram_gb"],
        "peak_gpu_gb":          run["peak_gpu_gb"],
        "auc":                  run.get("auc", float("nan")),
    }
    for j, k in enumerate(ks):
        fields[f"precision@{k}"] = run["precision"][j]
        fields[f"recall@{k}"]    = run["recall"][j]
        fields[f"ndcg@{k}"]      = run["ndcg"][j]
        fields[f"hit@{k}"]       = run["hit"][j]
        fields[f"mrr@{k}"]       = run["mrr"][j]
    return fields


def _print_run(run):
    ks = run.get("ks", [])
    rank_str = "  ".join(
        f"P@{k}={run['precision'][j]:.4f} R@{k}={run['recall'][j]:.4f} "
        f"NDCG@{k}={run['ndcg'][j]:.4f} Hit@{k}={run['hit'][j]:.4f} MRR@{k}={run['mrr'][j]:.4f}"
        for j, k in enumerate(ks)
    )
    print(
        f"  seed={run['seed']:5d} | "
        f"total={run['total_time_s']:7.1f}s "
        f"(train={run['train_time_s']:.1f}s  {run['time_per_epoch_s']:.3f}s/ep  "
        f"conv@ep{run['convergence_epoch']}  "
        f"eval={run['eval_time_s']:.1f}s  {run['inference_throughput']:.0f} users/s) | "
        f"ram={run['peak_ram_gb']:.2f}GB  gpu={run['peak_gpu_gb']:.2f}GB | "
        f"AUC={run.get('auc', float('nan')):.4f} | {rank_str}"
    )


def _print_summary(all_runs, model_name):
    if not all_runs:
        return {}
    all_scalars = [_scalar_fields(r) for r in all_runs]
    keys = list(all_scalars[0].keys())
    summary = {}
    for key in keys:
        vals = np.array([s[key] for s in all_scalars], dtype=float)
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    print(f"\n{'='*76}")
    print(f"  Mean ± Std  ({len(all_runs)} seeds)  —  {model_name.upper()} MIMIC-IV")
    print(f"{'='*76}")
    col_w = 22
    for key, ms in summary.items():
        print(f"  {key.ljust(col_w)}: {ms['mean']:>10.4f} ± {ms['std']:.4f}")
    return summary


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-seed MIMIC-IV benchmark for SIGN")
    parser.add_argument("--embeddings",  default=DEFAULT_PT,
                        help="Preprocessed .pt file from mimic_preprocessing.py")
    parser.add_argument("--epochs",      type=int,   default=45)
    parser.add_argument("--hidden",      type=int,   default=256,
                        help="MLP hidden channel width")
    parser.add_argument("--num-layers",  type=int,   default=3)
    parser.add_argument("--dropout",     type=float, default=0.0)
    parser.add_argument("--lr",          type=float, default=0.01)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--ks",          nargs="+",  type=int, default=DEFAULT_KS)
    parser.add_argument("--seeds",       nargs="+",  type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device",      type=int,   default=0)
    parser.add_argument("--out",         default="benchmark_results_sign.json")
    bargs = parser.parse_args()

    seeds = bargs.seeds
    device = torch.device(
        f"cuda:{bargs.device}" if torch.cuda.is_available() else "cpu"
    )

    # ── load preprocessed embeddings once ────────────────────────────────
    if not os.path.exists(bargs.embeddings):
        raise FileNotFoundError(
            f"'{bargs.embeddings}' not found. "
            "Run  python mimic_preprocessing.py  first."
        )
    print(f"Loading {bargs.embeddings} ...")
    try:
        op_dict = torch.load(bargs.embeddings, weights_only=False)
    except TypeError:
        op_dict = torch.load(bargs.embeddings)

    x           = torch.cat(op_dict["op_embedding"], dim=1)  # (n_users, total_feat_dim)
    y           = op_dict["labels"]                           # (n_users, n_items)
    n_users     = op_dict["n_users"]
    n_items     = op_dict["n_items"]
    train_items = op_dict["train_items"]
    test_set    = op_dict["test_set"]
    split_idx   = op_dict["split_idx"]
    train_idx   = split_idx["train"]
    val_idx     = split_idx["valid"]

    print(f"  n_users={n_users}  n_items={n_items}  "
          f"total_feat_dim={x.shape[-1]}  device={device}")

    shared = dict(
        x=x, y=y,
        train_idx=train_idx, val_idx=val_idx,
        train_items=train_items, test_set=test_set,
        n_users=n_users, n_items=n_items,
    )

    # ── banner ──────────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  SIGN MIMIC-IV Benchmark  |  seeds={seeds}  "
          f"epochs={bargs.epochs}  hidden={bargs.hidden}  ks={bargs.ks}")
    print(f"{'='*76}")

    # ── run each seed ────────────────────────────────────────────────────
    all_runs = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        result = run_one_seed(seed, shared, bargs, device)
        _print_run(result)
        all_runs.append(result)

    # ── summary ──────────────────────────────────────────────────────────
    summary = _print_summary(all_runs, "SIGN")

    # ── save JSON ─────────────────────────────────────────────────────────
    report = {
        "baseline": "sign",
        "epochs":   bargs.epochs,
        "hidden":   bargs.hidden,
        "ks":       bargs.ks,
        "seeds":    seeds,
        "n_users":  n_users,
        "n_items":  n_items,
        "runs":     all_runs,
        "summary":  summary,
    }
    with open(bargs.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {bargs.out}")


if __name__ == "__main__":
    main()
