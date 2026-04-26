#!/usr/bin/env python3
"""
Build MIMIC-style artifacts under Data/eICU from eICU patient.csv + diagnosis.csv.

Outputs (interaction + KG layout aligned with MIMIC-style baselines; user features are eICU-native):
  train2.txt, test2.txt, train.txt, test.txt (KGAT copies)
  feature.npz, patient_fix_features.csv, eicu_user_feature_legend.csv, mimic_target.csv
  kg_final0.txt  (user -> diagnosis-category, relation id 1)
  s_adj_mat2.npz, s_norm_adj_mat2.npz, s_mean_adj_mat2.npz  (via CDD Data loader)

Usage (from repo root):
  python scripts/build_eicu_dataset.py
  python scripts/build_eicu_dataset.py --num-items 2000 --test-ratio 0.2 --seed 42
  python scripts/build_eicu_dataset.py --max-diagnosis-rows 500000   # debug faster
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]


def _norm_field(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def _parse_age_bucket(age_val) -> int:
    """Return age bucket index 0..8 (not MIMIC column names)."""
    if pd.isna(age_val):
        return 8
    s = str(age_val).strip()
    if s.startswith(">"):
        return 8
    try:
        a = int(float(s))
    except ValueError:
        return 8
    if a <= 20:
        return 0
    if a <= 30:
        return 1
    if a <= 40:
        return 2
    if a <= 50:
        return 3
    if a <= 60:
        return 4
    if a <= 70:
        return 5
    if a <= 80:
        return 6
    if a <= 89:
        return 7
    return 8


AGE_BUCKET_LABELS = [
    "<=20",
    "21-30",
    "31-40",
    "41-50",
    "51-60",
    "61-70",
    "71-80",
    "81-89",
    ">=90_or_unknown",
]


def _build_eicu_user_feature_schema(
    patients: pd.DataFrame,
) -> tuple[list[str], list[tuple[str, str, str]], dict[str, int], dict[str, int]]:
    """
    User features = 9 age buckets + one-hot per distinct gender + one-hot per distinct ethnicity
    (all distinct values taken from full patient.csv, no mapping to MIMIC race_*).
    Returns: column_names, legend_rows, gender_to_idx, eth_to_idx
    """
    g_vocab = sorted({_norm_field(x) for x in patients["gender"].values})
    e_vocab = sorted({_norm_field(x) for x in patients["ethnicity"].values})

    col_names: list[str] = [f"age_{i}" for i in range(9)]
    legend: list[tuple[str, str, str]] = []
    for i in range(9):
        legend.append((str(i), "age", AGE_BUCKET_LABELS[i]))

    g_base = len(col_names)
    for j, g in enumerate(g_vocab):
        idx = g_base + j
        col_names.append(f"gender__{j}")
        legend.append((str(idx), "gender", g if g != "" else "(empty)"))

    e_base = len(col_names)
    for k, e in enumerate(e_vocab):
        idx = e_base + k
        col_names.append(f"ethnicity__{k}")
        legend.append((str(idx), "ethnicity", e if e != "" else "(empty)"))

    gender_to_idx = {g: g_base + j for j, g in enumerate(g_vocab)}
    eth_to_idx = {e: e_base + k for k, e in enumerate(e_vocab)}

    return col_names, legend, gender_to_idx, eth_to_idx


def extract_icd_token(raw) -> str | None:
    if pd.isna(raw):
        return None
    s = str(raw).strip().strip('"')
    if not s:
        return None
    first = s.split(",")[0].strip()
    if not first:
        return None
    return first.upper().replace(" ", "")


def first_diagnosis_category(diagnosisstring) -> str:
    if pd.isna(diagnosisstring):
        return "unknown"
    s = str(diagnosisstring).strip()
    if not s:
        return "unknown"
    return s.split("|")[0].strip().lower() or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eicu-dir",
        type=Path,
        default=REPO_ROOT / "Data" / "eICU",
        help="Directory containing patient.csv and diagnosis.csv",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "Data" / "eICU")
    ap.add_argument("--num-items", type=int, default=2000, help="Top-N ICD codes as items")
    ap.add_argument("--test-ratio", type=float, default=0.2, help="Per-user holdout for test items")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-diagnosis-rows", type=int, default=None, help="Limit rows for quick tests")
    ap.add_argument("--kg-categories", type=int, default=44, help="Number of diagnosis categories in kg tails (match mimic)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    eicu_dir = args.eicu_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_path = eicu_dir / "patient.csv"
    diag_path = eicu_dir / "diagnosis.csv"
    if not patient_path.is_file() or not diag_path.is_file():
        raise SystemExit(f"Need {patient_path} and {diag_path}")

    print("Loading patient.csv ...")
    patients = pd.read_csv(patient_path, low_memory=False)
    stay_to_row = {int(r["patientunitstayid"]): r for _, r in patients.iterrows()}

    col_names, legend_rows, gender_to_idx, eth_to_idx = _build_eicu_user_feature_schema(patients)
    n_gender_dim = len([r for r in legend_rows if r[1] == "gender"])
    n_eth_dim = len([r for r in legend_rows if r[1] == "ethnicity"])
    print(
        f"  eICU user features: 9 age + {n_gender_dim} gender + {n_eth_dim} ethnicity "
        f"= {len(col_names)} total dims"
    )
    print(f"  Distinct ethnicity field values in patient.csv (one dimension each): {n_eth_dim}")

    print("Loading diagnosis.csv (this may take a few minutes) ...")
    diag_iter = pd.read_csv(
        diag_path,
        usecols=["patientunitstayid", "icd9code", "diagnosisstring"],
        low_memory=False,
        chunksize=500_000,
    )

    icd_counts: Counter = Counter()
    chunks = []
    total = 0
    for chunk in diag_iter:
        if args.max_diagnosis_rows and total >= args.max_diagnosis_rows:
            break
        if args.max_diagnosis_rows:
            remain = args.max_diagnosis_rows - total
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain]
        chunk = chunk.copy()
        chunk["icd"] = chunk["icd9code"].map(extract_icd_token)
        chunk = chunk[chunk["icd"].notna()]
        chunks.append(chunk)
        icd_counts.update(chunk["icd"].tolist())
        total += len(chunk)
        if args.max_diagnosis_rows and total >= args.max_diagnosis_rows:
            break

    diag_df = pd.concat(chunks, ignore_index=True)
    print(f"  diagnosis rows used: {len(diag_df):,} (non-empty ICD)")

    top_icds = [c for c, _ in icd_counts.most_common(args.num_items)]
    icd_to_item = {c: i for i, c in enumerate(top_icds)}
    print(f"  top-{args.num_items} ICD codes selected.")

    diag_df = diag_df[diag_df["icd"].isin(icd_to_item)]
    diag_df["item_id"] = diag_df["icd"].map(icd_to_item)
    diag_df["cat"] = diag_df["diagnosisstring"].map(first_diagnosis_category)

    cat_counts = Counter(diag_df["cat"].tolist())
    top_cats = [c for c, _ in cat_counts.most_common(args.kg_categories)]
    cat_to_id = {c: i for i, c in enumerate(top_cats)}

    # user -> set of items and category set (for kg)
    user_items: dict[int, set[int]] = defaultdict(set)
    user_cats: dict[int, set[int]] = defaultdict(set)

    for uid_raw, iid_raw, cat_raw in zip(
        diag_df["patientunitstayid"].values,
        diag_df["item_id"].values,
        diag_df["cat"].values,
    ):
        uid = int(uid_raw)
        if uid not in stay_to_row:
            continue
        iid = int(iid_raw)
        user_items[uid].add(iid)
        cat = cat_raw if cat_raw in cat_to_id else top_cats[0]
        user_cats[uid].add(cat_to_id[cat])

    user_ids = sorted(user_items.keys())
    n_users = len(user_ids)
    stay_to_user = {s: u for u, s in enumerate(user_ids)}
    print(f"  users with >=1 interaction in top ICDs: {n_users:,}")

    # Per-user train/test split
    train_map: dict[int, list[int]] = {}
    test_map: dict[int, list[int]] = {}
    for uid in user_ids:
        items = sorted(user_items[uid])
        if len(items) < 2:
            train_map[uid] = items
            test_map[uid] = []
            continue
        n_test = max(1, int(round(len(items) * args.test_ratio)))
        n_test = min(n_test, len(items) - 1)
        perm = rng.permutation(len(items))
        test_idx = set(perm[:n_test])
        tr, te = [], []
        for j, it in enumerate(items):
            (te if j in test_idx else tr).append(it)
        if not te:
            te.append(tr.pop())
        train_map[uid] = sorted(tr)
        test_map[uid] = sorted(te)

    def write_cf_lines(path, tmap, label):
        lines = []
        for uid in sorted(tmap.keys()):
            items = tmap[uid]
            if not items:
                continue
            lines.append(
                " ".join([str(stay_to_user[uid])] + [str(float(x)) for x in items]) + "\n"
            )
        path.write_text("".join(lines), encoding="utf-8")
        print(f"  wrote {label}: {len(lines):,} lines -> {path}")

    train2_path = out_dir / "train2.txt"
    test2_path = out_dir / "test2.txt"
    write_cf_lines(train2_path, train_map, "train2")
    write_cf_lines(test2_path, test_map, "test2")

    # KGAT copies
    (out_dir / "train.txt").write_text(train2_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "test.txt").write_text(test2_path.read_text(encoding="utf-8"), encoding="utf-8")

    # kg_final0.txt: (user_idx, 1, category_id) unique pairs
    kg_lines = []
    for uid in user_ids:
        u = stay_to_user[uid]
        for c in user_cats[uid]:
            kg_lines.append(f"{u} 1 {c}\n")
    (out_dir / "kg_final0.txt").write_text("".join(kg_lines), encoding="utf-8")
    print(f"  wrote kg_final0.txt: {len(kg_lines):,} triples")

    # patient_fix_features.csv + feature.npz (eICU-native dims; see eicu_user_feature_legend.csv)
    feat_mat = np.zeros((n_users, len(col_names)), dtype=np.float32)

    for uid in user_ids:
        u = stay_to_user[uid]
        row = stay_to_row[uid]
        ab = _parse_age_bucket(row.get("age"))
        feat_mat[u, ab] = 1.0
        gkey = _norm_field(row.get("gender", ""))
        feat_mat[u, gender_to_idx[gkey]] = 1.0
        ekey = _norm_field(row.get("ethnicity", ""))
        feat_mat[u, eth_to_idx[ekey]] = 1.0

    pdf = pd.DataFrame(feat_mat, columns=col_names)
    pdf_path = out_dir / "patient_fix_features.csv"
    pdf.to_csv(pdf_path, index=False)
    print(f"  wrote {pdf_path} shape={feat_mat.shape}")

    legend_path = out_dir / "eicu_user_feature_legend.csv"
    pd.DataFrame(legend_rows, columns=["feature_index", "group", "value"]).to_csv(
        legend_path, index=False
    )
    print(f"  wrote {legend_path}")

    sp.save_npz(out_dir / "feature.npz", sp.csr_matrix(feat_mat))
    print(f"  wrote {out_dir / 'feature.npz'}")

    # mimic_target.csv: first train item per user (or 0)
    target_rows = []
    for uid in user_ids:
        u = stay_to_user[uid]
        tr = train_map.get(uid, [])
        tgt = int(tr[0]) if tr else 0
        target_rows.append({"id": u, "target": tgt})
    pd.DataFrame(target_rows).to_csv(out_dir / "mimic_target.csv", index=False)
    print(f"  wrote mimic_target.csv")

    # Adjacency matrices (same as CDD Data)
    for adj_name in ("s_adj_mat2.npz", "s_norm_adj_mat2.npz", "s_mean_adj_mat2.npz"):
        p = out_dir / adj_name
        if p.is_file():
            p.unlink()

    sys.path.insert(0, str(REPO_ROOT / "CDD"))
    from utility.load_data import Data  # noqa: E402

    print("Building adjacency matrices (first run may take a while) ...")
    dg = Data(path=str(out_dir), batch_size=1024)
    dg.get_adj_mat()
    print("Done.")

    kgat_ds = REPO_ROOT / "KGAT" / "datasets" / "eICU"
    kgat_ds.mkdir(parents=True, exist_ok=True)
    for name in (
        "train2.txt",
        "test2.txt",
        "train.txt",
        "test.txt",
        "kg_final0.txt",
        "feature.npz",
        "eicu_user_feature_legend.csv",
    ):
        src = out_dir / name
        if src.is_file():
            dst = kgat_ds / name
            dst.write_bytes(src.read_bytes())
    print(f"Copied CF + KG + feature.npz to {kgat_ds}")

    print("\nNext: run models with dataset path Data/eICU (see scripts/run_eicu_baselines.ps1).")


if __name__ == "__main__":
    main()
