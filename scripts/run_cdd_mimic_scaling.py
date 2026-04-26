import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt


def _read_cf(path: Path):
    user_items = {}
    max_uid = -1
    max_iid = -1
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            uid = int(float(parts[0]))
            items = [int(float(x)) for x in parts[1:]]
            user_items[uid] = items
            max_uid = max(max_uid, uid)
            if items:
                max_iid = max(max_iid, max(items))
    return user_items, max_uid, max_iid


def _write_cf(path: Path, mapping, user_items):
    lines = []
    n_edges = 0
    for old_uid in sorted(mapping.keys()):
        items = user_items.get(old_uid, [])
        if not items:
            continue
        new_uid = mapping[old_uid]
        n_edges += len(items)
        lines.append(" ".join([str(new_uid)] + [str(float(i)) for i in items]))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return n_edges


def main():
    ap = argparse.ArgumentParser(description="CDD scaling benchmark on nested MIMIC-IV patient subsets.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fractions", default="50,75,100")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--embed-size", type=int, default=64)
    ap.add_argument("--layer-size", default="[64,64,64]")
    ap.add_argument("--test-cores", type=int, default=1)
    ap.add_argument("--run", action="store_true", help="Run CDD training after preparing subsets.")
    ap.add_argument("--data-dir", default="Data/mimicIV")
    ap.add_argument("--subset-root", default="Data/mimicIV_nested_subsets")
    ap.add_argument("--output", default="rebuttal/cdd_mimic_scaling_summary.tsv")
    ap.add_argument("--plot", default="rebuttal/cdd_time_vs_edges_loglog.png")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = (repo_root / args.data_dir).resolve()
    subset_root = (repo_root / args.subset_root).resolve()
    output_path = (repo_root / args.output).resolve()
    plot_path = (repo_root / args.plot).resolve()
    cdd_dir = (repo_root / "CDD").resolve()

    train_path = data_dir / "train2.txt"
    test_path = data_dir / "test2.txt"
    feat_path = data_dir / "feature.npz"
    if not train_path.is_file() or not test_path.is_file() or not feat_path.is_file():
        raise SystemExit("Missing train2.txt/test2.txt/feature.npz under Data/mimicIV.")

    train_map, train_max_u, train_max_i = _read_cf(train_path)
    test_map, test_max_u, test_max_i = _read_cf(test_path)
    n_users = max(train_max_u, test_max_u) + 1
    n_items = max(train_max_i, test_max_i) + 1
    active_users = sorted(set(train_map.keys()) | set(test_map.keys()))

    features = sp.load_npz(str(feat_path)).tocsr()
    if features.shape[0] < n_users:
        raise SystemExit(f"feature.npz rows ({features.shape[0]}) < n_users ({n_users}).")

    rng = np.random.default_rng(args.seed)
    perm_active = rng.permutation(len(active_users))
    fracs = [int(x.strip()) for x in args.fractions.split(",") if x.strip()]

    rows = []
    for pct in fracs:
        # Sample from active users only so train/test-derived n_users matches feature rows.
        active_take = int(round(len(active_users) * (pct / 100.0)))
        active_take = max(1, min(len(active_users), active_take))
        selected_idx = perm_active[:active_take]
        selected = np.array(sorted([active_users[i] for i in selected_idx]), dtype=np.int64)
        take = len(selected)
        mapping = {int(old): int(new) for new, old in enumerate(selected.tolist())}
        subset_name = f"mimicIV_p{pct}"
        subset_dir = subset_root / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        n_train_edges = _write_cf(subset_dir / "train2.txt", mapping, train_map)
        n_test_edges = _write_cf(subset_dir / "test2.txt", mapping, test_map)
        n_edges = n_train_edges + n_test_edges

        # CDD also reads train.txt/test.txt in some scripts; keep copies.
        (subset_dir / "train.txt").write_text((subset_dir / "train2.txt").read_text(encoding="utf-8"), encoding="utf-8")
        (subset_dir / "test.txt").write_text((subset_dir / "test2.txt").read_text(encoding="utf-8"), encoding="utf-8")

        sp.save_npz(str(subset_dir / "feature.npz"), features[selected])

        run_time_per_epoch = np.nan
        peak_gpu_mb = np.nan
        throughput = np.nan

        if args.run:
            weights_dir = (repo_root / "trained_model" / "CDD_scaling" / subset_name)
            weights_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python",
                "main.py",
                "--dataset", subset_name,
                "--data_path", str(subset_root) + os.sep,
                "--epoch", str(args.epochs),
                "--gpu_id", str(args.gpu_id),
                "--weights_path", str(weights_dir) + os.sep,
                "--batch_size", str(args.batch_size),
                "--embed_size", str(args.embed_size),
                "--layer_size", str(args.layer_size),
                "--verbose", "10",
            ]
            print("Running:", " ".join(cmd))
            env = os.environ.copy()
            env["CDD_TEST_CORES"] = str(args.test_cores)
            env["CDD_SKIP_DETAILED_REPORT"] = "1"
            subprocess.run(cmd, cwd=str(cdd_dir), check=True, env=env)

            eff_path = weights_dir / "efficiency.tsv"
            if eff_path.is_file():
                eff = pd.read_csv(eff_path, sep="\t")
                run_time_per_epoch = float(eff["avg_epoch_time_s"].iloc[0])
                peak_gpu_mb = float(eff["peak_gpu_mb"].iloc[0])
                throughput = float(eff["inference_throughput_users_s"].iloc[0])

        rows.append({
            "subset": subset_name,
            "fraction_pct": pct,
            "n_users_selected": take,
            "n_items": n_items,
            "edges_E": n_edges,
            "train_edges": n_train_edges,
            "test_edges": n_test_edges,
            "train_time_per_epoch_s": run_time_per_epoch,
            "peak_gpu_mb": peak_gpu_mb,
            "inference_throughput_patients_s": throughput,
        })

    out_df = pd.DataFrame(rows).sort_values("fraction_pct")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved summary: {output_path}")
    print(out_df.to_string(index=False))

    if out_df["train_time_per_epoch_s"].notna().sum() >= 2:
        p = out_df.dropna(subset=["train_time_per_epoch_s"]).sort_values("edges_E")
        plt.figure(figsize=(6.4, 4.2))
        plt.loglog(p["edges_E"].to_numpy(), p["train_time_per_epoch_s"].to_numpy(), marker="o")
        plt.xlabel("|E| (number of edges)")
        plt.ylabel("training time per epoch (s)")
        plt.title("CDD Scaling on nested MIMIC-IV subsets")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()

