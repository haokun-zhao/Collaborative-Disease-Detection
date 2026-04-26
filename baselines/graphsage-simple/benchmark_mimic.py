"""
Multi-seed MIMIC-IV benchmark for GraphSAGE-simple.

Metrics logged per seed:
  Ranking  : AUC, Precision, Recall, NDCG, Hit Rate, MRR  @  K ∈ {3,5,10,20}
  Timing   : total_time_s, train_time_s, eval_time_s,
             time_per_epoch_s (mean), inference_throughput (users/s)
  Convergence: train_log (train loss + epoch time per epoch) saved to JSON
  Memory   : peak_ram_gb, peak_gpu_gb

Usage (from graphsage-simple/):
    python benchmark_mimic.py
    python benchmark_mimic.py --seeds 42 123 456
    python benchmark_mimic.py --epochs 50 --embed-dim 128

NOTE: Each seed re-randomises both model init and the 90/10 train/val user split.
"""

import sys, os, time, json, random, argparse, gc
import numpy as np
import torch

try:
    import psutil
    _has_psutil = True
except ImportError:
    _has_psutil = False

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_DATA = _ROOT / "Data" / "mimicIV"

sys.path.insert(0, str(_HERE))
from graphsage.model import load_mimic, MimicGraphSage, _rank_metrics
from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
import torch.nn as nn

DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_KS    = [3, 5, 10, 20]


# ── resource helpers ──────────────────────────────────────────────────────────

def _peak_ram_gb():
    if not _has_psutil:
        return float("nan")
    mem = psutil.Process().memory_info()
    return getattr(mem, "peak_wset", mem.rss) / 1024 ** 3

def _peak_gpu_gb():
    return torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0

def _reset_gpu():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ── single-seed run ──────────────────────────────────────────────────────────

def run_one_seed(seed, shared, args):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    _reset_gpu()

    feat_data    = shared["feat_data"]
    multi_hot    = shared["multi_hot"]
    adj_lists    = shared["adj_lists"]
    n_users      = shared["n_users"]
    n_items      = shared["n_items"]
    n_user_feats = shared["n_user_feats"]
    train_items  = shared["train_items"]
    test_set     = shared["test_set"]
    ks_list      = args.ks
    num_nodes    = n_users + n_items
    feat_dim     = n_user_feats

    # ── model ──────────────────────────────────────────────────────────
    features_emb = nn.Embedding(num_nodes, feat_dim)
    features_emb.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features_emb, cuda=False)
    enc1 = Encoder(features_emb, feat_dim, args.embed_dim, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, args.embed_dim, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = args.num_samples
    enc2.num_samples = args.num_samples
    graphsage = MimicGraphSage(n_items, enc2)

    # ── train/val split ── only users with training interactions ──────
    all_train_users = sorted(train_items.keys())
    random.shuffle(all_train_users)
    n_val = max(1, int(0.1 * len(all_train_users)))
    train_sup_users = all_train_users[n_val:]

    labels_tensor = torch.FloatTensor(multi_hot)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=args.lr
    )

    # ── training ─────────────────────────────────────────────────────────
    train_log = []
    t_train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        _ep_t0 = time.perf_counter()
        graphsage.train()
        batch = random.sample(train_sup_users, min(args.batch_size, len(train_sup_users)))
        optimizer.zero_grad()
        loss = graphsage.loss(batch, labels_tensor[batch])
        if not torch.isfinite(loss):
            continue           # skip NaN/Inf batch; don't corrupt weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graphsage.parameters(), max_norm=5.0)
        optimizer.step()
        ep_time = time.perf_counter() - _ep_t0
        train_log.append({
            "epoch":        epoch,
            "train_loss":   round(float(loss.item()), 6),
            "epoch_time_s": round(ep_time, 4),
        })
        if epoch % 10 == 0:
            print(f"    epoch {epoch}/{args.epochs}  loss={loss.item():.4f}  time={ep_time:.3f}s")
    train_time_s = time.perf_counter() - t_train_start

    # ── evaluation ───────────────────────────────────────────────────────
    t_eval_start = time.perf_counter()
    graphsage.eval()
    with torch.no_grad():
        chunk = 2048
        all_scores = np.concatenate([
            graphsage.forward(list(range(s, min(s + chunk, n_users)))).numpy()
            for s in range(0, n_users, chunk)
        ], axis=0)
    eval_time_s = time.perf_counter() - t_eval_start

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

    time_per_epoch_s      = train_time_s / args.epochs
    inference_throughput  = round(n_users / eval_time_s, 1) if eval_time_s > 0 else 0

    del graphsage, enc1, enc2, agg1, agg2, features_emb, all_scores
    gc.collect()

    return {
        "seed":                  seed,
        "train_time_s":          round(train_time_s, 3),
        "eval_time_s":           round(eval_time_s,  3),
        "total_time_s":          round(train_time_s + eval_time_s, 3),
        "time_per_epoch_s":      round(time_per_epoch_s, 4),
        "epochs_run":            args.epochs,
        "convergence_epoch":     None,   # no val-based early stopping in GraphSAGE
        "inference_throughput":  inference_throughput,
        "peak_ram_gb":           round(_peak_ram_gb(), 3),
        "peak_gpu_gb":           round(_peak_gpu_gb(), 3),
        "n_eval":                n_ev,
        "auc":                   float(sum_auc / n_ev) if n_ev else float("nan"),
        "precision":             [float(v / n_ev) for v in sum_prec] if n_ev else [],
        "recall":                [float(v / n_ev) for v in sum_rec]  if n_ev else [],
        "ndcg":                  [float(v / n_ev) for v in sum_ndcg] if n_ev else [],
        "hit":                   [float(v / n_ev) for v in sum_hit]  if n_ev else [],
        "mrr":                   [float(v / n_ev) for v in sum_mrr]  if n_ev else [],
        "ks":                    ks_list,
        "train_log":             train_log,
    }


# ── aggregation & reporting ───────────────────────────────────────────────────

def _scalar_fields(run):
    ks = run.get("ks", [])
    f = {
        "total_time_s":         run["total_time_s"],
        "train_time_s":         run["train_time_s"],
        "eval_time_s":          run["eval_time_s"],
        "time_per_epoch_s":     run["time_per_epoch_s"],
        "epochs_run":           run["epochs_run"],
        "inference_throughput": run["inference_throughput"],
        "peak_ram_gb":          run["peak_ram_gb"],
        "peak_gpu_gb":          run["peak_gpu_gb"],
        "auc":                  run.get("auc", float("nan")),
    }
    for j, k in enumerate(ks):
        f[f"precision@{k}"] = run["precision"][j]
        f[f"recall@{k}"]    = run["recall"][j]
        f[f"ndcg@{k}"]      = run["ndcg"][j]
        f[f"hit@{k}"]       = run["hit"][j]
        f[f"mrr@{k}"]       = run["mrr"][j]
    return f


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
        f"eval={run['eval_time_s']:.1f}s  {run['inference_throughput']:.0f} users/s) | "
        f"ram={run['peak_ram_gb']:.2f}GB  gpu={run['peak_gpu_gb']:.2f}GB | "
        f"AUC={run.get('auc', float('nan')):.4f} | {rank_str}"
    )


def _print_summary(all_runs, model_name):
    if not all_runs:
        return {}
    scalars = [_scalar_fields(r) for r in all_runs]
    keys    = list(scalars[0].keys())
    summary = {}
    for key in keys:
        vals = np.array([s[key] for s in scalars], dtype=float)
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
    parser = argparse.ArgumentParser(description="Multi-seed MIMIC-IV benchmark for GraphSAGE")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=0.01)
    parser.add_argument("--embed-dim",   type=int,   default=128)
    parser.add_argument("--num-samples", type=int,   default=5)
    parser.add_argument("--ks",          nargs="+",  type=int, default=DEFAULT_KS)
    parser.add_argument("--seeds",       nargs="+",  type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--data-dir",    default=str(_DATA))
    parser.add_argument("--out",         default="benchmark_results_graphsage.json")
    bargs = parser.parse_args()

    seeds = bargs.seeds

    print(f"Loading MIMIC-IV data from {bargs.data_dir} ...")
    (feat_data, multi_hot, adj_lists, n_users, n_items,
     n_user_feats, train_items, test_set) = load_mimic(bargs.data_dir)
    print(f"  n_users={n_users}  n_items={n_items}  feat_dim={n_user_feats}  "
          f"total_nodes={n_users + n_items}")

    shared = dict(
        feat_data=feat_data, multi_hot=multi_hot, adj_lists=adj_lists,
        n_users=n_users, n_items=n_items, n_user_feats=n_user_feats,
        train_items=train_items, test_set=test_set,
    )

    class _Args:
        pass
    run_args = _Args()
    run_args.epochs      = bargs.epochs
    run_args.batch_size  = bargs.batch_size
    run_args.lr          = bargs.lr
    run_args.embed_dim   = bargs.embed_dim
    run_args.num_samples = bargs.num_samples
    run_args.ks          = bargs.ks

    print(f"\n{'='*76}")
    print(f"  GraphSAGE MIMIC-IV Benchmark  |  seeds={seeds}  "
          f"epochs={bargs.epochs}  ks={bargs.ks}")
    print(f"{'='*76}")

    all_runs = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        result = run_one_seed(seed, shared, run_args)
        _print_run(result)
        all_runs.append(result)

    summary = _print_summary(all_runs, "GraphSAGE")

    report = {
        "baseline": "graphsage",
        "epochs":   bargs.epochs,
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
