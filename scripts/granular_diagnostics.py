import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import os


def _read_interactions_txt(path: Path) -> Dict[int, List[int]]:
    m: Dict[int, List[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            uid = int(float(parts[0]))
            items = [int(float(x)) for x in parts[1:]]
            m[uid] = items
    return m


def _disease_train_freq(train_items: Dict[int, List[int]], n_items: int) -> np.ndarray:
    freq = np.zeros(n_items, dtype=np.int64)
    for _, items in train_items.items():
        for it in items:
            if 0 <= it < n_items:
                freq[it] += 1
    return freq


def _split_head_mid_tail(n_items: int, train_freq: np.ndarray) -> Dict[str, Set[int]]:
    # Sort by descending frequency, break ties by item id (stable).
    order = np.lexsort((np.arange(n_items), -train_freq))
    # lexsort sorts ascending; we want descending freq: use -freq then id, but above yields correct:
    # Actually lexsort uses last key as primary. Here keys=(id, -freq): primary -freq (descending), tiebreak id.
    ranked = order

    n_head = int(round(0.20 * n_items))
    n_mid = int(round(0.30 * n_items))
    n_tail = n_items - n_head - n_mid

    head = set(map(int, ranked[:n_head]))
    mid = set(map(int, ranked[n_head : n_head + n_mid]))
    tail = set(map(int, ranked[n_head + n_mid :]))
    assert len(head) + len(mid) + len(tail) == n_items
    return {"Head": head, "Mid": mid, "Tail": tail}


def _ndcg_at_k_binary(topk: Sequence[int], rel_set: Set[int], k: int) -> float:
    # DCG
    dcg = 0.0
    for i, it in enumerate(topk[:k]):
        if it in rel_set:
            dcg += 1.0 / np.log2(i + 2.0)
    # IDCG: best case puts all relevant first
    ideal_hits = min(len(rel_set), k)
    if ideal_hits == 0:
        return float("nan")
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else float("nan")


def _per_user_group_metrics(
    detailed_rows: List[dict],
    k: int,
    item_groups: Dict[str, Set[int]],
) -> pd.DataFrame:
    out_rows = []
    for r in detailed_rows:
        uid = int(r["user_id"])
        test_items = set(map(int, r.get("test_items", [])))
        if not test_items:
            continue
        topk = list(map(int, r.get("top_predicted", [])))[:k]
        train_cnt = len(r.get("training_items", []))

        for gname, gitems in item_groups.items():
            gt = test_items.intersection(gitems)
            if not gt:
                continue
            hits = sum(1 for it in topk if it in gt)
            recall = hits / float(len(gt)) if len(gt) > 0 else float("nan")
            ndcg = _ndcg_at_k_binary(topk, gt, k=k)
            out_rows.append(
                {
                    "user_id": uid,
                    "group": gname,
                    "train_interactions": int(train_cnt),
                    "test_group_size": int(len(gt)),
                    "hits@k": int(hits),
                    "recall@k": float(recall),
                    "ndcg@k": float(ndcg),
                }
            )
    return pd.DataFrame(out_rows)


def _summarize_groups(per_user_df: pd.DataFrame) -> pd.DataFrame:
    if per_user_df.empty:
        return pd.DataFrame(columns=["group", "n_users", "recall@20", "ndcg@20"])
    g = per_user_df.groupby("group", as_index=False).agg(
        n_users=("user_id", "nunique"),
        recall_at_k=("recall@k", "mean"),
        ndcg_at_k=("ndcg@k", "mean"),
    )
    g = g.rename(columns={"recall_at_k": "recall@20", "ndcg_at_k": "ndcg@20"})
    return g


def _history_density_groups(train_items: Dict[int, List[int]]) -> Tuple[Dict[str, Set[int]], Dict[str, int]]:
    counts = np.array([len(v) for _, v in sorted(train_items.items())], dtype=np.int64)
    if counts.size == 0:
        return {"Sparse": set(), "Medium": set(), "Dense": set()}, {"p33": 0, "p66": 0}
    p33 = int(np.percentile(counts, 33))
    p66 = int(np.percentile(counts, 66))
    sparse, medium, dense = set(), set(), set()
    for u, items in train_items.items():
        c = len(items)
        if c <= p33:
            sparse.add(int(u))
        elif c <= p66:
            medium.add(int(u))
        else:
            dense.add(int(u))
    return {"Sparse": sparse, "Medium": medium, "Dense": dense}, {"p33": p33, "p66": p66}


def _load_ccsr_map(ccsr_csv: Path) -> Dict[int, str]:
    """
    Use CDD/icd10_codes_all.csv which already contains a per-item ccsr_category (3-letter body system).
    Columns: index, ..., ccsr_category
    """
    df = pd.read_csv(ccsr_csv)
    if "index" not in df.columns or "ccsr_category" not in df.columns:
        raise ValueError(f"{ccsr_csv}: expected columns 'index' and 'ccsr_category'")
    m = {}
    for _, row in df.iterrows():
        try:
            idx = int(row["index"])
        except Exception:
            continue
        cat = str(row["ccsr_category"]).strip()
        if not cat or cat.lower() == "nan":
            continue
        m[idx] = cat
    return m


def _ccsr_groups_from_test_edges(
    test_set: Dict[int, List[int]],
    item_to_ccsr: Dict[int, str],
    top_n: int,
    coverage: float,
) -> Tuple[Dict[str, Set[int]], pd.DataFrame]:
    # Count test edges by category
    counts: Dict[str, int] = {}
    for _, items in test_set.items():
        for it in items:
            cat = item_to_ccsr.get(int(it), "Other")
            counts[cat] = counts.get(cat, 0) + 1
    total = sum(counts.values()) if counts else 0
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    chosen = []
    covered = 0
    for cat, c in ranked:
        if cat == "Other":
            continue
        if len(chosen) < top_n and (total == 0 or (covered / total) < coverage):
            chosen.append(cat)
            covered += c
        else:
            # still might need to meet coverage even if exceeding top_n; keep it strict to top_n as requested
            continue

    chosen_set = set(chosen)
    groups: Dict[str, Set[int]] = {cat: set() for cat in chosen}
    groups["Other"] = set()
    for it, cat in item_to_ccsr.items():
        if cat in chosen_set:
            groups[cat].add(int(it))
        else:
            groups["Other"].add(int(it))

    meta = pd.DataFrame(
        [
            {
                "ccsr": cat,
                "test_edges": int(c),
                "share": (float(c) / float(total)) if total > 0 else 0.0,
            }
            for cat, c in ranked
        ]
    )
    return groups, meta


def _eval_cldd_detailed(data_dir: Path, checkpoint: Path, gpu_id: int) -> List[dict]:
    # Use CDD's existing globals (like fairness_eval does).
    cdd_dir = data_dir.parent.parent / "CDD"
    # Ensure we import from CDD/ as a package-local run
    import sys as _sys
    if str(cdd_dir) not in _sys.path:
        _sys.path.insert(0, str(cdd_dir))

    # CDD.utility.batch_test calls parse_args() at import time. Provide only the
    # args it needs (dataset + data_path + batch_size) and hide our script flags.
    saved_argv = list(_sys.argv)
    _sys.argv = [
        _sys.argv[0],
        "--dataset",
        data_dir.name,
        "--data_path",
        str(data_dir.parent) + os.sep,
        "--batch_size",
        "1024",
        "--Ks",
        "[20]",
    ]
    from utility.batch_test import data_generator as dg, test as cdd_test  # type: ignore
    from utility.batch_test import args as cdd_args  # type: ignore
    _sys.argv = saved_argv
    from CDD import CDD as CDDModel  # type: ignore

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    cdd_args.gpu_id = gpu_id
    cdd_args.dataset = data_dir.name
    cdd_args.data_path = str(data_dir.parent) + os.sep
    cdd_args.device = device
    # Ensure dropout args are numeric lists (parser defaults are strings like "[0.1,0.1,0.1]").
    if isinstance(getattr(cdd_args, "node_dropout", None), str):
        cdd_args.node_dropout = eval(cdd_args.node_dropout)
    if isinstance(getattr(cdd_args, "mess_dropout", None), str):
        cdd_args.mess_dropout = eval(cdd_args.mess_dropout)

    _, norm_adj, _ = dg.get_adj_mat()
    feature_matrix = sp.load_npz(str(data_dir / "feature.npz"))
    model = CDDModel(dg.n_users, dg.n_items, norm_adj, feature_matrix, cdd_args).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    users_to_test = list(dg.test_set.keys())
    _, detailed = cdd_test(model, users_to_test, drop_flag=False, batch_test_flag=False, return_detailed=True)
    return detailed


def _eval_lightgcn_detailed(data_dir: Path, checkpoint: Path, gpu_id: int, device: str = "auto") -> List[dict]:
    # Run within LightGCN folder import path
    lgcn_dir = data_dir.parent.parent / "LightGCN"
    import sys as _sys
    if str(lgcn_dir) not in _sys.path:
        _sys.path.insert(0, str(lgcn_dir))

    from LightGCN import LightGCN as LightGCNModel  # type: ignore

    # utility.batch_test parses args at import time; emulate minimal argv like export script.
    import sys
    sys.argv = [
        sys.argv[0],
        "--dataset",
        data_dir.name,
        "--data_path",
        str(data_dir.parent) + os.sep,
        "--batch_size",
        "1024",
        "--Ks",
        "[20]",
    ]
    import utility.batch_test as bt  # type: ignore

    # Resolve device: reuse export_per_patient_metrics behavior is overkill; keep simple.
    if device == "cpu":
        dev = torch.device("cpu")
    elif device == "cuda":
        dev = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    class _Args:
        pass

    margs = _Args()
    margs.device = dev
    margs.embed_size = 64
    margs.layer_size = "[64,64,64]"
    margs.regs = "[1e-4]"
    margs.batch_size = 1024

    _, norm_adj, _ = bt.data_generator.get_adj_mat()
    model = LightGCNModel(bt.data_generator.n_users, bt.data_generator.n_items, norm_adj, margs).to(dev)
    ckpt = torch.load(checkpoint, map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    users_to_test = sorted(list(bt.data_generator.test_set.keys()))
    _, detailed = bt.test(model, users_to_test, drop_flag=False, return_detailed=True)
    return detailed


def _eval_ngcf_detailed(
    repo_root: Path,
    data_name: str,
    data_dir_root: Path,
    checkpoint: Path,
    gpu_id: int,
    k: int,
) -> List[dict]:
    """
    Evaluate NGCF and return per-user rows containing:
      user_id, training_items, test_items, top_predicted (top-k after filtering training items)
    Uses KGAT's DataLoaderBPRMF + model.NGCF.
    """
    import sys as _sys

    kgat_dir = (repo_root / "KGAT").resolve()
    if str(kgat_dir) not in _sys.path:
        _sys.path.insert(0, str(kgat_dir))

    from data_loader.loader_bprmf import DataLoaderBPRMF  # type: ignore
    from model.NGCF import NGCF  # type: ignore
    from main_ngcf import build_norm_adj  # type: ignore
    from utils.model_helper import load_model  # type: ignore

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    class A:
        pass

    a = A()
    a.seed = 42
    a.data_name = data_name
    a.data_dir = str(data_dir_root)
    a.use_pretrain = 0
    a.pretrain_embedding_dir = str((repo_root / "KGAT" / "datasets" / "pretrain").resolve())
    a.pretrain_model_path = ""
    a.embed_dim = 64
    a.l2loss_lambda = 1e-5
    a.train_batch_size = 1024
    a.test_batch_size = 10000
    a.lr = 1e-4
    a.n_epoch = 1
    a.stopping_steps = 10
    a.print_every = 999999
    a.evaluate_every = 1
    a.Ks = f"[{int(k)}]"
    a.batch_size = 1024
    a.layer_size = "[64,64,64]"
    a.regs = "[1e-5]"
    a.mess_dropout = "[0.1,0.1,0.1]"
    a.node_dropout = "[0.1]"
    a.node_dropout_flag = 0
    a.adj_type = "norm"
    a.alg_type = "ngcf"
    a.save_dir = ""

    data = DataLoaderBPRMF(a, logging=type("L", (), {"info": lambda *_: None})())
    norm_adj = build_norm_adj(data.n_users, data.n_items, data.train_user_dict)
    model = NGCF(a, data.n_users, data.n_items, norm_adj)
    model = load_model(model, str(checkpoint))
    model.to(device)
    model.eval()

    detailed: List[dict] = []
    with torch.no_grad():
        u_emb, i_emb = model.get_propagated_embeddings()
        i_emb_t = i_emb.t()
        for uid in sorted(data.test_user_dict.keys()):
            train_items = list(map(int, data.train_user_dict.get(uid, [])))
            test_items = list(map(int, data.test_user_dict.get(uid, [])))
            if not test_items:
                continue
            u = torch.tensor([uid], dtype=torch.long, device=device)
            scores = torch.matmul(u_emb[u], i_emb_t).view(-1).detach().cpu().numpy()

            # Filter training items from ranking candidates (same convention as other evaluators).
            if train_items:
                scores[np.array(train_items, dtype=np.int64)] = -np.inf
            topk = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
            topk = topk[np.argsort(-scores[topk])]

            detailed.append(
                {
                    "user_id": int(uid),
                    "training_items": train_items,
                    "test_items": test_items,
                    "top_predicted": list(map(int, topk.tolist())),
                }
            )
    return detailed


def _compare_tables(cldd: pd.DataFrame, base: pd.DataFrame, key_col: str = "group") -> pd.DataFrame:
    m = cldd.merge(base, on=key_col, how="outer", suffixes=("_cldd", "_base"))
    for metric in ["recall@20", "ndcg@20"]:
        m[f"{metric}_rel_impr_%"] = (m[f"{metric}_cldd"] - m[f"{metric}_base"]) / m[f"{metric}_base"] * 100.0
    return m


def main():
    ap = argparse.ArgumentParser(description="Granular diagnostic analyses for CLDD gains.")
    ap.add_argument("--data-dir", default="Data/mimicIV", help="Dataset directory containing train2.txt/test2.txt/feature.npz.")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--cldd-checkpoint", required=True, help="CLDD checkpoint (.pkl saved by CDD/main.py).")
    ap.add_argument("--baseline", default="ngcf", choices=("lightgcn", "ngcf"))
    ap.add_argument("--lightgcn-checkpoint", default="", help="LightGCN checkpoint (.pkl). Used when --baseline=lightgcn.")
    ap.add_argument("--ngcf-checkpoint", default="KGAT/trained_model/NGCF/mimicIV/embed-dim64_layer-size[64,64,64]_lr0.0001_pretrain2_seed456/model_epoch390.pth", help="NGCF checkpoint (.pth). Used when --baseline=ngcf.")
    ap.add_argument("--ngcf-data-dir", default="KGAT/datasets/", help="Root dir that contains <data-name>/train.txt,test.txt,... Used when --baseline=ngcf.")
    ap.add_argument("--ngcf-data-name", default="mimicIV", help="Dataset folder name under --ngcf-data-dir. Usually mimicIV or eICU.")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--ccsr-map", default="CDD/icd10_codes_all.csv", help="Path to item_id->ccsr_category CSV.")
    ap.add_argument("--ccsr-top-n", type=int, default=10)
    ap.add_argument("--ccsr-coverage", type=float, default=0.90)
    ap.add_argument("--out-prefix", default="rebuttal/granular", help="Output prefix for TSVs.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    train = _read_interactions_txt(data_dir / "train2.txt")
    test = _read_interactions_txt(data_dir / "test2.txt")
    n_items = 0
    for m in (train, test):
        for _, items in m.items():
            if items:
                n_items = max(n_items, max(items))
    n_items = n_items + 1

    # Model detailed results
    cldd_det = _eval_cldd_detailed(data_dir, Path(args.cldd_checkpoint), args.gpu_id)
    if args.baseline == "lightgcn":
        if not args.lightgcn_checkpoint:
            raise SystemExit("--lightgcn-checkpoint is required when --baseline=lightgcn")
        base_det = _eval_lightgcn_detailed(data_dir, Path(args.lightgcn_checkpoint), args.gpu_id)
        baseline_name = "LightGCN"
    else:
        if not args.ngcf_checkpoint:
            raise SystemExit("--ngcf-checkpoint is required when --baseline=ngcf")
        repo_root = Path(__file__).resolve().parents[1]
        base_det = _eval_ngcf_detailed(
            repo_root=repo_root,
            data_name=args.ngcf_data_name,
            data_dir_root=(repo_root / args.ngcf_data_dir).resolve(),
            checkpoint=Path(args.ngcf_checkpoint).resolve(),
            gpu_id=args.gpu_id,
            k=args.k,
        )
        baseline_name = "NGCF"

    # ── Analysis 1: prevalence head/mid/tail ─────────────────────────────────
    freq = _disease_train_freq(train, n_items)
    hmt = _split_head_mid_tail(n_items, freq)
    cldd_hmt = _summarize_groups(_per_user_group_metrics(cldd_det, args.k, hmt))
    base_hmt = _summarize_groups(_per_user_group_metrics(base_det, args.k, hmt))
    hmt_cmp = _compare_tables(cldd_hmt, base_hmt, key_col="group")
    hmt_cmp.insert(0, "baseline", baseline_name)

    # ── Analysis 2: clinical group (CCSR body system) ────────────────────────
    ccsr_map = _load_ccsr_map((repo_root / args.ccsr_map).resolve())
    ccsr_groups, ccsr_meta = _ccsr_groups_from_test_edges(test, ccsr_map, args.ccsr_top_n, args.ccsr_coverage)
    cldd_ccsr = _summarize_groups(_per_user_group_metrics(cldd_det, args.k, ccsr_groups))
    base_ccsr = _summarize_groups(_per_user_group_metrics(base_det, args.k, ccsr_groups))
    ccsr_cmp = _compare_tables(cldd_ccsr, base_ccsr, key_col="group")
    ccsr_cmp.insert(0, "baseline", baseline_name)

    # ── Analysis 3: patient history density ──────────────────────────────────
    dens_groups, dens_thr = _history_density_groups(train)
    # Convert user-based groups into item_groups-like interface by filtering users after metric calc:
    # reuse per_user_group_metrics on a single "AllItems" group, then stratify by user density.
    all_items = set(range(n_items))
    per_user_cldd = _per_user_group_metrics(cldd_det, args.k, {"All": all_items})
    per_user_base = _per_user_group_metrics(base_det, args.k, {"All": all_items})

    def _summ_by_density(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for name, users in dens_groups.items():
            sub = df[df["user_id"].isin(users)]
            # Only keep users that have at least 1 test item (already ensured by per_user_group_metrics)
            if sub.empty:
                continue
            rows.append(
                {
                    "group": name,
                    "n_users": int(sub["user_id"].nunique()),
                    "recall@20": float(sub["recall@k"].mean()),
                    "ndcg@20": float(sub["ndcg@k"].mean()),
                }
            )
        return pd.DataFrame(rows)

    cldd_dens = _summ_by_density(per_user_cldd)
    base_dens = _summ_by_density(per_user_base)
    dens_cmp = _compare_tables(cldd_dens, base_dens, key_col="group")
    dens_cmp.insert(0, "baseline", baseline_name)

    out_prefix = (repo_root / args.out_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    hmt_cmp.to_csv(str(out_prefix) + "_prevalence.tsv", sep="\t", index=False)
    ccsr_cmp.to_csv(str(out_prefix) + "_ccsr.tsv", sep="\t", index=False)
    ccsr_meta.to_csv(str(out_prefix) + "_ccsr_meta.tsv", sep="\t", index=False)
    dens_cmp.to_csv(str(out_prefix) + "_density.tsv", sep="\t", index=False)
    pd.DataFrame([dens_thr]).to_csv(str(out_prefix) + "_density_thresholds.tsv", sep="\t", index=False)

    print("Saved:")
    print("  " + str(out_prefix) + "_prevalence.tsv")
    print("  " + str(out_prefix) + "_ccsr.tsv")
    print("  " + str(out_prefix) + "_density.tsv")


if __name__ == "__main__":
    main()

