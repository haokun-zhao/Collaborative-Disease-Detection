"""
R1-Q6: Influence attribution for CLDD predictions via Integrated Gradients.

Uses path integral from a baseline z0 (default: all-zero initial embedding) to the
actual initial embedding z, with an n-step Riemann sum (default 50). Completeness:
    sum(IG) ≈ score(z) - score(z0)
using CDD.forward_from_init() (see CDD/CDD.py).

Per history disease d': sum over dimensions of IG row for node (patient+d').
Demographic slice: sum of IG over the patient row's last f dims (fixed features).

Outputs:
  - Case-level CSV: Patient A top-5 targets, top-3 historical contributors each + demo %
  - Aggregate: median % of marginal score in top-3 contributors; median % demographic
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

# Globals set in main (needed by compute_influence_scores / predict_all)
TRAIN_ITEMS: Dict[int, List[int]] = {}
DISEASE_NAME_BY_ITEM: Dict[int, str] = {}
MODEL: torch.nn.Module = None  # type: ignore

# Set in main(): Integrated Gradients hyperparameters
_IG_CONFIG: Dict[str, object] = {"n_steps": 50, "baseline": "zero"}


def _read_interactions(path: Path) -> Dict[int, List[int]]:
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


def load_disease_names(repo_root: Path) -> Dict[int, str]:
    """
    Map item id (0..n_items-1) -> short label: ICD code + title if available.
    Uses CDD/icd10_codes_all.csv + CDD/d_icd_diagnoses.csv.
    """
    codes_path = repo_root / "CDD" / "icd10_codes_all.csv"
    icd_path = repo_root / "CDD" / "d_icd_diagnoses.csv"
    if not codes_path.exists():
        return {}

    def norm_code(c: str) -> str:
        return str(c).replace(".", "").strip().upper()

    title_by_code: Dict[str, str] = {}
    if icd_path.exists():
        ddf = pd.read_csv(icd_path, dtype=str)
        for _, row in ddf.iterrows():
            code = norm_code(row.get("icd_code", ""))
            if code:
                title_by_code[code] = str(row.get("long_title", code))[:120]

    cdf = pd.read_csv(codes_path)
    out: Dict[int, str] = {}
    for _, row in cdf.iterrows():
        try:
            idx = int(row["index"])
        except Exception:
            continue
        code_raw = str(row.get("icd10_code", row.get("original_code", ""))).strip()
        nk = norm_code(code_raw)
        title = title_by_code.get(nk, "")
        if title:
            out[idx] = f"{code_raw} — {title}"
        else:
            out[idx] = code_raw or f"item_{idx}"
    return out


def load_cldd_model(
    checkpoint_path: Path,
    data_path: str,
    dataset: str,
    gpu_id: int,
) -> torch.nn.Module:
    """Load trained CLDD; attaches n_patients, k, f, predict_all for this script."""
    repo_root = Path(__file__).resolve().parents[1]
    cdd_dir = repo_root / "CDD"
    if str(cdd_dir) not in sys.path:
        sys.path.insert(0, str(cdd_dir))

    dp = data_path if data_path.endswith(os.sep) else data_path + os.sep
    saved_argv = list(sys.argv)
    sys.argv = [
        saved_argv[0],
        "--dataset",
        dataset,
        "--data_path",
        dp,
        "--batch_size",
        "1024",
        "--Ks",
        "[20]",
    ]
    from utility.batch_test import data_generator as dg  # type: ignore
    from utility.batch_test import args as cdd_args  # type: ignore
    from CDD import CDD as CDDModel  # type: ignore

    sys.argv = saved_argv

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    cdd_args.gpu_id = gpu_id
    cdd_args.dataset = dataset
    cdd_args.data_path = dp
    if isinstance(getattr(cdd_args, "node_dropout", None), str):
        cdd_args.node_dropout = eval(cdd_args.node_dropout)
    if isinstance(getattr(cdd_args, "mess_dropout", None), str):
        cdd_args.mess_dropout = eval(cdd_args.mess_dropout)

    _, norm_adj, _ = dg.get_adj_mat()
    feat_path = Path(dp.rstrip(os.sep)) / dataset / "feature.npz"
    feature_matrix = sp.load_npz(str(feat_path))
    model = CDDModel(dg.n_users, dg.n_items, norm_adj, feature_matrix, cdd_args).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Attributes expected by compute_influence_scores()
    model.n_patients = model.n_user
    model.k = model.emb_size + model.feature.shape[1]
    model.f = int(model.feature.shape[1])

    def predict_all(patient_id: int) -> torch.Tensor:
        return model.predict_all_scores(int(patient_id), TRAIN_ITEMS.get(int(patient_id), []), drop_flag=False)

    model.predict_all = predict_all  # type: ignore[assignment]
    return model


def get_neighbors(p: int) -> List[int]:
    return list(TRAIN_ITEMS.get(int(p), []))


def get_disease_name(d: int) -> str:
    if d < 0:
        return "N/A"
    return DISEASE_NAME_BY_ITEM.get(int(d), f"disease_id={int(d)}")


def build_z_baseline(model: torch.nn.Module) -> torch.Tensor:
    """Initial-embedding baseline x0 (same shape / device / dtype as z_real)."""
    z_ref = model.get_initial_embeddings()
    kind = str(_IG_CONFIG.get("baseline", "zero"))
    if kind == "zero":
        return torch.zeros_like(z_ref)
    raise ValueError(f"Unknown IG baseline: {kind!r} (supported: 'zero')")


def integrated_gradients(
    model: torch.nn.Module,
    patient_id: int,
    target_disease: int,
    z_init_baseline: torch.Tensor,
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Riemann sum along z(alpha) = z0 + alpha * (z_real - z0), alpha in {1/n, ..., 1}.
    Returns IG tensor of same shape as z_real; completeness: IG.sum() ≈ score(z_real)-score(z0).
    """
    z_real = model.get_initial_embeddings().clone().detach()
    z0 = z_init_baseline.to(device=z_real.device, dtype=z_real.dtype)
    diff = z_real - z0
    total_grad = torch.zeros_like(z_real)
    alphas = torch.linspace(
        1.0 / n_steps, 1.0, n_steps, device=z_real.device, dtype=z_real.dtype
    )
    pid = int(patient_id)
    tid = int(target_disease)
    npat = int(model.n_patients)

    for alpha in alphas:
        z_interp = z0 + alpha * diff
        z_interp = z_interp.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            z_final = model.forward_from_init(z_interp, drop_flag=False)
            z_p = z_final[pid]
            z_d_final = z_final[npat + tid]
            score = torch.dot(z_p, z_d_final)
            grad = torch.autograd.grad(
                score, z_interp, retain_graph=False, create_graph=False
            )[0]
        total_grad = total_grad + grad / float(n_steps)

    return diff * total_grad


def _score_at_z_init(model, patient_id: int, target_disease_id: int, z_init: torch.Tensor) -> float:
    with torch.no_grad():
        zf = model.forward_from_init(z_init, drop_flag=False)
        s = torch.dot(
            zf[int(patient_id)],
            zf[int(model.n_patients) + int(target_disease_id)],
        )
        return float(s.item())


def _pct_denom(score: float, baseline_score: float) -> float:
    """Prefer marginal score for IG percentages so shares align with completeness."""
    d = score - baseline_score
    if abs(d) > 1e-6:
        return d
    return abs(score) if abs(score) > 1e-6 else 1.0


def compute_influence_scores(model, patient_id: int, target_disease_id: int):
    neighbors = get_neighbors(patient_id)
    z_real = model.get_initial_embeddings().clone().detach()
    z0 = build_z_baseline(model)
    n_steps = int(_IG_CONFIG.get("n_steps", 50))

    ig = integrated_gradients(
        model, patient_id, target_disease_id, z0, n_steps=n_steps
    )

    score = _score_at_z_init(model, patient_id, target_disease_id, z_real)
    baseline_score = _score_at_z_init(model, patient_id, target_disease_id, z0)
    ig_sum = float(ig.sum().item())
    completeness_gap = ig_sum - (score - baseline_score)
    denom = _pct_denom(score, baseline_score)

    per_disease = {}
    for d_prime in neighbors:
        idx = model.n_patients + d_prime
        per_disease[d_prime] = float(ig[idx].sum().item())

    k = int(model.k)
    f = int(model.f)
    demographic_infl = float(ig[patient_id, k - f :].sum().item())

    return {
        "score": score,
        "baseline_score": baseline_score,
        "ig_sum": ig_sum,
        "completeness_gap": completeness_gap,
        "denom": denom,
        "per_disease": per_disease,
        "demographic": demographic_infl,
    }


def _pad_top_contribs(per_disease: Dict[int, float], top_contributors: int):
    items = sorted(per_disease.items(), key=lambda x: -x[1])[:top_contributors]
    while len(items) < top_contributors:
        items.append((-1, 0.0))
    return items


def case_level_analysis(model, patient_id: int, top_k: int = 5, top_contributors: int = 3) -> pd.DataFrame:
    with torch.no_grad():
        all_scores = model.predict_all(patient_id)
        top_diseases = torch.topk(all_scores, top_k).indices.tolist()

    rows = []
    for rank, d in enumerate(top_diseases, 1):
        result = compute_influence_scores(model, patient_id, d)
        score = result["score"]
        baseline_score = result["baseline_score"]
        denom = result["denom"]
        per_disease = result["per_disease"]
        demo = result["demographic"]

        top_contribs = _pad_top_contribs(per_disease, top_contributors)
        top_sum = sum(v for (di, v) in top_contribs if di >= 0)
        top_pct = 100.0 * top_sum / denom if denom != 0 else 0.0
        demo_pct = 100.0 * demo / denom if denom != 0 else 0.0

        def fmt_contrib(tup):
            di, val = tup
            if di < 0 or denom == 0:
                return "N/A"
            return f"{get_disease_name(di)} ({100 * val / denom:.1f}%)"

        rows.append(
            {
                "rank": rank,
                "target_disease": d,
                "target_name": get_disease_name(d),
                "score": score,
                "baseline_score": baseline_score,
                "ig_sum": result["ig_sum"],
                "completeness_gap": result["completeness_gap"],
                "top_contributor_1": fmt_contrib(top_contribs[0]),
                "top_contributor_2": fmt_contrib(top_contribs[1]),
                "top_contributor_3": fmt_contrib(top_contribs[2]),
                "top3_pct": top_pct,
                "demo_pct": demo_pct,
            }
        )
    return pd.DataFrame(rows)


def aggregate_analysis(
    model,
    patient_ids: List[int],
    top_k: int = 5,
    top_contributors: int = 3,
    sample: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    pids = list(patient_ids)
    if sample is not None and sample < len(pids):
        rng = np.random.default_rng(42)
        pids = rng.choice(pids, size=sample, replace=False).tolist()

    all_top3_pct = []
    all_demo_pct = []
    for p in pids:
        with torch.no_grad():
            all_scores = model.predict_all(int(p))
            top_diseases = torch.topk(all_scores, top_k).indices.tolist()
        for d in top_diseases:
            result = compute_influence_scores(model, int(p), int(d))
            denom = result["denom"]
            if denom == 0:
                continue
            per_disease = result["per_disease"]
            demo = result["demographic"]
            vals = sorted(per_disease.values(), reverse=True)[:top_contributors]
            top_sum = sum(vals)
            all_top3_pct.append(100 * top_sum / denom)
            all_demo_pct.append(100 * demo / denom)

    return np.array(all_top3_pct), np.array(all_demo_pct)


def main():
    ap = argparse.ArgumentParser(description="CLDD influence attribution (R1-Q6).")
    ap.add_argument("--checkpoint", type=Path, required=True, help="CLDD .pkl checkpoint (same as Tables 1–2).")
    ap.add_argument("--data-path", default="Data/", help="Data root ending with / or without (e.g. Data/).")
    ap.add_argument("--dataset", default="mimicIV")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--patient-a", type=int, default=9260, help="Case study patient (Table 1 highest precision).")
    ap.add_argument("--out-dir", type=Path, default=Path("rebuttal"))
    ap.add_argument(
        "--aggregate-sample",
        type=int,
        default=None,
        help="If set, subsample this many test patients for aggregate stats; default = all test patients.",
    )
    ap.add_argument("--ig-steps", type=int, default=50, help="Integrated Gradients Riemann steps.")
    ap.add_argument(
        "--ig-baseline",
        default="zero",
        choices=("zero",),
        help="Baseline initial embedding x0. 'zero' is all zeros (simplest; may be OOD).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dp_arg = Path(args.data_path)
    data_root = dp_arg.resolve() if dp_arg.is_absolute() else (repo_root / dp_arg).resolve()
    data_dir = data_root / args.dataset
    train_path = data_dir / "train2.txt"
    test_path = data_dir / "test2.txt"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(f"Missing {train_path} or {test_path}")

    global TRAIN_ITEMS, DISEASE_NAME_BY_ITEM, MODEL, _IG_CONFIG
    _IG_CONFIG = {"n_steps": int(args.ig_steps), "baseline": str(args.ig_baseline)}
    TRAIN_ITEMS = _read_interactions(train_path)
    test_map = _read_interactions(test_path)
    test_patient_ids = sorted(test_map.keys())

    DISEASE_NAME_BY_ITEM = load_disease_names(repo_root)

    dp = str(data_root) + os.sep
    MODEL = load_cldd_model(args.checkpoint.resolve(), dp, args.dataset, args.gpu_id)
    model = MODEL

    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_df = case_level_analysis(model, args.patient_a)
    case_path = args.out_dir / "case_level_patient_a.csv"
    case_df.to_csv(case_path, index=False)
    print("=== CASE-LEVEL (Patient A) ===")
    print(case_df.to_string(index=False))
    print(f"\nSaved: {case_path}")

    sample = args.aggregate_sample
    top3_pct, demo_pct = aggregate_analysis(model, test_patient_ids, sample=sample)
    n_pairs = len(top3_pct)
    print("\n=== AGGREGATE ===")
    if sample is None:
        print(f"Patients: {len(test_patient_ids)} (all test); (patient, top-5) pairs: {n_pairs}")
    else:
        print(f"Subsampled patients: {sample}; pairs: {n_pairs}")
    print(f"Median % of score in top-3 contributors:  {np.median(top3_pct):.1f}%")
    print(f"Mean   % of score in top-3 contributors:  {np.mean(top3_pct):.1f}%")
    print(f"Fraction with >50% in top-3:               {np.mean(top3_pct > 50):.1%}")
    print(f"Median % from demographic prior:           {np.median(demo_pct):.1f}%")
    print(f"Mean   % from demographic prior:           {np.mean(demo_pct):.1f}%")

    summ_path = args.out_dir / "influence_aggregate_summary.tsv"
    pd.DataFrame(
        [
            {
                "n_test_patients": len(test_patient_ids),
                "aggregate_sample": sample if sample is not None else len(test_patient_ids),
                "n_pairs_evaluated": n_pairs,
                "median_top3_contrib_pct": float(np.median(top3_pct)) if n_pairs else float("nan"),
                "median_demo_pct": float(np.median(demo_pct)) if n_pairs else float("nan"),
            }
        ]
    ).to_csv(summ_path, sep="\t", index=False)
    print(f"Saved: {summ_path}")


if __name__ == "__main__":
    main()
