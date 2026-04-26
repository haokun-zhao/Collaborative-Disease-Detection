"""
Multi-seed MIMIC-IV benchmark for MixHop / N-GCN.

Metrics logged per seed:
  Ranking  : AUC, Precision, Recall, NDCG, Hit Rate, MRR  @  K ∈ {3,5,10,20}
  Timing   : total_time_s, train_time_s, eval_time_s,
             time_per_epoch_s (mean), inference_throughput (users/s)
  Convergence: convergence_epoch (epoch of best val-BCE), val loss curve saved to JSON
  Memory   : peak_ram_gb, peak_gpu_gb

Usage (from MixHop-and-N-GCN/src/):
    python benchmark_mimic.py                       # mixhop, 5 seeds
    python benchmark_mimic.py --model ngcn
    python benchmark_mimic.py --seeds 42 123 456
    python benchmark_mimic.py --epochs 100 --mimic-hidden 64
"""

import sys, os, time, json, copy, random, argparse
import numpy as np
import torch
import pandas as pd

try:
    import psutil
    _has_psutil = True
except ImportError:
    _has_psutil = False

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_DATA = _ROOT / "Data" / "mimicIV"

from param_parser import parameter_parser
from trainer_and_networks import Trainer
from utils import (
    load_mimic_propagator,
    load_mimic_train_test,
    load_mimic_features_from_npz,
    load_mimic_target_full,
)

DEFAULT_SEEDS = [42]
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

def run_one_seed(seed, base_args, shared, per_patient_dir=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    _reset_gpu()

    args = copy.copy(base_args)
    args.seed               = seed
    args.n_users            = shared["n_users"]
    args.n_items            = shared["n_items"]
    args.propagation_matrix = shared["propagation_matrix"]
    args.train_items        = shared["train_items"]
    args.test_set           = shared["test_set"]

    t0 = time.perf_counter()
    trainer = Trainer(args, None, shared["features"], shared["target"], True)
    trainer.fit()
    t_train = time.perf_counter()

    if per_patient_dir:
        metrics, per_user = trainer.evaluate_mimic_ranking(return_per_user=True)
        if per_user:
            os.makedirs(per_patient_dir, exist_ok=True)
            pd.DataFrame(per_user).to_csv(
                os.path.join(per_patient_dir, f"mixhop_per_patient_seed{seed}.csv"),
                index=False
            )
    else:
        metrics = trainer.evaluate_mimic_ranking()   # returns full dict including inference info
    t_total = time.perf_counter()

    if not metrics:   # handles None (old trainer) or empty dict (n_ev==0)
        metrics = {}

    train_time_s  = t_train - t0
    eval_time_s   = t_total - t_train
    total_time_s  = t_total - t0
    epochs_run    = getattr(trainer, "epochs_trained", args.epochs)
    conv_epoch    = getattr(trainer, "convergence_epoch", epochs_run)
    train_log     = getattr(trainer, "train_log", [])
    time_per_ep   = float(np.mean([e["epoch_time_s"] for e in train_log])) if train_log else 0.0
    inf_tp        = metrics.get("inference_throughput", 0)

    return {
        "seed":                   seed,
        "train_time_s":           round(train_time_s, 3),
        "eval_time_s":            round(eval_time_s,  3),
        "total_time_s":           round(total_time_s, 3),
        "time_per_epoch_s":       round(time_per_ep, 4),
        "epochs_run":             epochs_run,
        "convergence_epoch":      conv_epoch,
        "inference_throughput":   inf_tp,
        "peak_ram_gb":            round(_peak_ram_gb(), 3),
        "peak_gpu_gb":            round(_peak_gpu_gb(), 3),
        "n_eval":                 metrics.get("n_eval", 0),
        "auc":                    metrics.get("auc", float("nan")),
        "precision":              metrics.get("precision", []),
        "recall":                 metrics.get("recall", []),
        "ndcg":                   metrics.get("ndcg", []),
        "hit":                    metrics.get("hit", []),
        "mrr":                    metrics.get("mrr", []),
        "ks":                     metrics.get("ks", []),
        "train_log":              train_log,   # full val-loss curve for convergence analysis
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
        "convergence_epoch":    run["convergence_epoch"],
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
        f"conv@ep{run['convergence_epoch']}  "
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
    parser = argparse.ArgumentParser(description="Multi-seed MIMIC-IV benchmark for MixHop/N-GCN")
    parser.add_argument("--model",          default="mixhop", choices=["mixhop", "ngcn"])
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--early-stopping", type=int,   default=10)
    parser.add_argument("--learning-rate",  type=float, default=0.01)
    parser.add_argument("--dropout",        type=float, default=0.5)
    parser.add_argument("--layers-1",       nargs="+",  type=int, default=[200, 200, 200])
    parser.add_argument("--layers-2",       nargs="+",  type=int, default=[200, 200, 200])
    parser.add_argument("--mimic-hidden",   type=int,   default=None)
    parser.add_argument("--ks",             nargs="+",  type=int, default=DEFAULT_KS)
    parser.add_argument("--seeds",          nargs="+",  type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--data-dir",       default=str(_DATA))
    parser.add_argument("--out",            default=None)
    parser.add_argument("--per-patient-dir", default=None,
                        help="Optional directory to save per-patient CSV per seed.")
    bargs = parser.parse_args()

    out_path = bargs.out or f"benchmark_results_{bargs.model}.json"
    seeds    = bargs.seeds

    sys.argv = ["benchmark_mimic.py", "--dataset", "mimic"]
    base_args = parameter_parser()
    base_args.model          = bargs.model
    base_args.epochs         = bargs.epochs
    base_args.early_stopping = bargs.early_stopping
    base_args.learning_rate  = bargs.learning_rate
    base_args.dropout        = bargs.dropout
    base_args.layers_1       = bargs.layers_1
    base_args.layers_2       = bargs.layers_2
    base_args.mimic_hidden   = bargs.mimic_hidden
    base_args.ks             = bargs.ks
    base_args.metric_every   = 0
    base_args.force_gpu      = True
    base_args.data_dir       = bargs.data_dir
    if base_args.mimic_hidden is not None:
        h = base_args.mimic_hidden
        base_args.layers_1 = [h, h, h]
        base_args.layers_2 = [h, h, h]

    data_dir = Path(base_args.data_dir)
    print(f"Loading MIMIC-IV data from {data_dir} ...")
    train_items, test_set, n_users, n_items = load_mimic_train_test(
        str(data_dir / "train2.txt"), str(data_dir / "test2.txt")
    )
    propagation_matrix = load_mimic_propagator(str(data_dir / "s_norm_adj_mat2.npz"))
    features = load_mimic_features_from_npz(str(data_dir / "feature.npz"), n_users, n_items)
    target   = load_mimic_target_full(str(data_dir / "mimic_target.csv"), n_users, n_items)
    print(f"  n_users={n_users}  n_items={n_items}  feat_dim={features['dimensions'][1]}")

    shared = dict(
        n_users=n_users, n_items=n_items,
        propagation_matrix=propagation_matrix,
        features=features, target=target,
        train_items=train_items, test_set=test_set,
    )

    print(f"\n{'='*76}")
    print(f"  {bargs.model.upper()} MIMIC-IV Benchmark  |  seeds={seeds}  "
          f"epochs={bargs.epochs}  ks={bargs.ks}")
    print(f"{'='*76}")

    all_runs = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        result = run_one_seed(seed, base_args, shared, bargs.per_patient_dir)
        _print_run(result)
        all_runs.append(result)

    summary = _print_summary(all_runs, bargs.model)

    report = {
        "baseline": bargs.model,
        "epochs":   bargs.epochs,
        "ks":       bargs.ks,
        "seeds":    seeds,
        "n_users":  n_users,
        "n_items":  n_items,
        "runs":     all_runs,
        "summary":  summary,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
