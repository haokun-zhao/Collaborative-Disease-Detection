import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


BEST_RE = re.compile(
    r"Best Iter=\[(?P<iter>\d+)\].*?recall=\[(?P<rec>[^\]]+)\], precision=\[(?P<pre>[^\]]+)\], "
    r"hit=\[(?P<hit>[^\]]+)\], ndcg=\[(?P<ndcg>[^\]]+)\], mrr=\[(?P<mrr>[^\]]+)\], auc=\[(?P<auc>[^\]]+)\]"
)


def _parse_vec(s):
    return [float(x) for x in re.split(r"[\t,\s]+", s.strip()) if x]


def _parse_best_metrics(log_text):
    m = BEST_RE.search(log_text)
    if not m:
        return {}
    rec = _parse_vec(m.group("rec"))
    pre = _parse_vec(m.group("pre"))
    hit = _parse_vec(m.group("hit"))
    ndcg = _parse_vec(m.group("ndcg"))
    mrr = _parse_vec(m.group("mrr"))
    auc = _parse_vec(m.group("auc"))
    out = {"best_iter": int(m.group("iter"))}
    for j, k in enumerate([3, 5, 10, 20]):
        if j < len(rec):
            out[f"recall@{k}"] = rec[j]
        if j < len(pre):
            out[f"precision@{k}"] = pre[j]
        if j < len(hit):
            out[f"hit@{k}"] = hit[j]
        if j < len(ndcg):
            out[f"ndcg@{k}"] = ndcg[j]
        if j < len(mrr):
            out[f"mrr@{k}"] = mrr[j]
    if auc:
        out["auc"] = auc[0]
    return out


def main():
    ap = argparse.ArgumentParser(description="Systematic CDD ablation runner with fixed tuning budget.")
    ap.add_argument("--dataset", default="mimicIV")
    ap.add_argument("--data-path", default="Data/")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--embed-size", type=int, default=64)
    ap.add_argument("--layer-size", default="[64,64,64]")
    ap.add_argument("--weights-root", default="trained_model/CDD_ablation")
    ap.add_argument("--output", default="rebuttal/cdd_ablation_results.tsv")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cdd_dir = repo_root / "CDD"
    weights_root = (repo_root / args.weights_root).resolve()
    output_path = (repo_root / args.output).resolve()
    weights_root.mkdir(parents=True, exist_ok=True)

    # Five XXX rows in the rebuttal table.
    # (E) "Weighted sum" (inter_layer_agg=mean) already has values; omitted here.
    # (F) embed_size and layer_size are changed together so hidden dims match the
    #     embedding dimension, keeping the ablation clean.
    variants = [
        # ("F_k32",  {"embed_size": "32",  "layer_size": "[32,32,32]"}),
        # ("F_k128", {"embed_size": "128", "layer_size": "[128,128,128]"}),
        # ("E_last_layer_only",     {"inter_layer_agg": "last"}),
        # ("D_wo_bilinear_branch",  {"aggregator_mode": "sum"}),
        ("C_uniform_hop_weights", {"hop_mixing": "uniform"}),
    ]

    seeds = [42]
    rows = []
    for name, extra in variants:
        seed_rows = []
        for seed in seeds:
            run_dir = weights_root / name / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python", "main.py",
                "--dataset",        args.dataset,
                "--data_path",      str((repo_root / args.data_path).resolve()) + os.sep,
                "--epoch",          str(args.epochs),
                "--seed",           str(seed),
                "--gpu_id",         str(args.gpu_id),
                "--batch_size",     str(args.batch_size),
                "--lr",             str(args.lr),
                "--embed_size",     str(args.embed_size),
                "--layer_size",     str(args.layer_size),
                "--Ks",             "[3,5,10,20]",
                "--weights_path",   str(run_dir) + os.sep,
                "--verbose",        "10",
                "--save_flag",      "1",
                "--max_hop",        "3",
                "--hop_mixing",     "adaptive",
                "--aggregator_mode","sum_bi",
                "--inter_layer_agg","concat",
                "--use_demographics","1",
            ]
            for k, v in extra.items():
                flag = f"--{k}"
                if flag in cmd:
                    i = cmd.index(flag)
                    cmd[i + 1] = v
                else:
                    cmd.extend([flag, v])

            print(f"\n=== Ablation: {name} | seed={seed} ===")
            print(" ".join(cmd))
            env = os.environ.copy()
            env["CDD_TEST_CORES"] = "1"
            env["CDD_SKIP_DETAILED_REPORT"] = "1"
            try:
                proc = subprocess.run(cmd, cwd=str(cdd_dir), capture_output=True, text=True, env=env, check=True)
                log_text = proc.stdout + "\n" + proc.stderr
            except subprocess.CalledProcessError as e:
                log_text = (e.stdout or "") + "\n" + (e.stderr or "")
                (run_dir / "run.log").write_text(log_text, encoding="utf-8")
                print(f"\n[ERROR] Ablation failed: {name} | seed={seed}")
                print(log_text[-4000:])
                raise
            (run_dir / "run.log").write_text(log_text, encoding="utf-8")

            best = _parse_best_metrics(log_text)
            seed_rows.append(best)

        if not seed_rows:
            continue
        numeric_keys = sorted({k for r in seed_rows for k in r.keys() if k != "best_iter"})
        means = {k: float(np.mean([r.get(k, np.nan) for r in seed_rows])) for k in numeric_keys}

        row = {
            "variant":          name,
            "dataset":          args.dataset,
            "seeds":            args.seeds,
            "epochs_budget":    args.epochs,
            "batch_size":       args.batch_size,
            "lr":               args.lr,
            "embed_size":       int(extra.get("embed_size", args.embed_size)),
            "layer_size":       extra.get("layer_size", args.layer_size),
            "max_hop":          int(extra.get("max_hop", 3)),
            "hop_mixing":       extra.get("hop_mixing", "adaptive"),
            "aggregator_mode":  extra.get("aggregator_mode", "sum_bi"),
            "inter_layer_agg":  extra.get("inter_layer_agg", "concat"),
            "use_demographics": int(extra.get("use_demographics", 1)),
        }
        row.update(means)
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"\nSaved ablation table: {output_path}")
    print(df.to_string(index=False))
    if not df.empty:
        print("\nTable-fill values (AUC, Recall@20, Precision@20, NDCG@20, Hit@20):")
        for _, r in df.iterrows():
            print(
                f"{r['variant']}: "
                f"{r.get('auc', np.nan):.4f}, {r.get('recall@20', np.nan):.4f}, "
                f"{r.get('precision@20', np.nan):.4f}, {r.get('ndcg@20', np.nan):.4f}, "
                f"{r.get('hit@20', np.nan):.4f}"
            )


if __name__ == "__main__":
    main()

