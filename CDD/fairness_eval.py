"""
fairness_eval.py — Subgroup fairness evaluation for CDD.

Evaluates precision, recall, NDCG, MRR, and hit-ratio at K=[5,10,15,20]
for demographic subgroups (age, sex, race) derived from patient_fix_features.csv.

Disparity measures reported per axis:
  - max_gap   : max(metric) - min(metric) across subgroups
  - min/max   : disparate-impact ratio (min / max)

Usage (from CDD/):
    python fairness_eval.py [--weights_path model/] [--checkpoint <file.pkl>]
    All other flags are inherited from utility/parser.py.
"""

import os
import glob
import sys
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')

from CDD import CDD
from utility.helper import *
from utility.batch_test import data_generator, test, Ks, args   # reuse globals


# ── Broad race buckets ───────────────────────────────────────────────────────
RACE_BUCKETS = {
    'White':    lambda c: c.startswith('race_WHITE') or c == 'race_PORTUGUESE',
    'Black':    lambda c: c.startswith('race_BLACK'),
    'Hispanic': lambda c: c.startswith('race_HISPANIC'),
    'Asian':    lambda c: c.startswith('race_ASIAN'),
    'Native':   lambda c: c in ('race_AMERICAN INDIAN/ALASKA NATIVE',
                                 'race_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'),
    'Other':    lambda c: c in ('race_MULTIPLE RACE/ETHNICITY', 'race_OTHER',
                                 'race_SOUTH AMERICAN'),
    'Unknown':  lambda c: c in ('race_PATIENT DECLINED TO ANSWER',
                                 'race_UNABLE TO OBTAIN', 'race_UNKNOWN'),
}


def load_demographics(feat_path, n_users):
    """Return a DataFrame with columns: user_id, age_group, sex, race_broad."""
    df = pd.read_csv(feat_path, dtype=str)   # read as strings to handle True/False

    # Align row count with model users (some rows may exceed n_users)
    df = df.iloc[:n_users].reset_index(drop=True)
    df.index.name = 'user_id'

    bool_df = df.apply(lambda col: col.str.lower() == 'true')

    age_cols  = [c for c in bool_df.columns if c.startswith('age_')]
    sex_cols  = [c for c in bool_df.columns if c.startswith('gender_')]
    race_cols = [c for c in bool_df.columns if c.startswith('race_')]

    records = []
    for uid in range(len(bool_df)):
        row = bool_df.iloc[uid]

        # Age: take first True age bucket; else 'Unknown'
        age = next((c.replace('age_', '') for c in age_cols if row[c]), 'Unknown')

        # Sex
        if row.get('gender_F', False):
            sex = 'F'
        elif row.get('gender_M', False):
            sex = 'M'
        else:
            sex = 'Unknown'

        # Race: map granular columns → broad bucket
        active_races = [c for c in race_cols if row[c]]
        race = 'Unknown'
        for bucket, pred in RACE_BUCKETS.items():
            if any(pred(c) for c in active_races):
                race = bucket
                break

        records.append({'user_id': uid, 'age_group': age, 'sex': sex, 'race': race})

    return pd.DataFrame(records)


def eval_subgroups(model, demo_df, axis_col, test_user_set):
    """For each subgroup in `axis_col`, compute metrics over test users in that group."""
    results = []
    groups = demo_df[axis_col].unique()

    for group in sorted(groups):
        group_users = demo_df.loc[demo_df[axis_col] == group, 'user_id'].tolist()
        test_users  = [u for u in group_users if u in test_user_set]
        n = len(test_users)
        if n == 0:
            continue

        ret = test(model, test_users, drop_flag=False)

        row = {axis_col: group, 'n_users': n}
        for ki, k in enumerate(Ks):
            row[f'precision@{k}'] = float(ret['precision'][ki])
            row[f'recall@{k}']    = float(ret['recall'][ki])
            row[f'ndcg@{k}']      = float(ret['ndcg'][ki])
            row[f'mrr@{k}']       = float(ret['mrr'][ki])
            row[f'hit@{k}']       = float(ret['hit_ratio'][ki])
        row['auc'] = float(ret['auc'])
        results.append(row)

    return pd.DataFrame(results)


def disparity_summary(df, group_col, metric_cols):
    """Return max_gap and min/max ratio for each metric column."""
    rows = []
    for m in metric_cols:
        vals = df[m].values
        if len(vals) < 2:
            continue
        max_v, min_v = vals.max(), vals.min()
        rows.append({
            'metric':    m,
            'max_gap':   float(max_v - min_v),
            'min/max':   float(min_v / max_v) if max_v > 0 else float('nan'),
            'max_group': df.loc[df[m].idxmax(), group_col],
            'min_group': df.loc[df[m].idxmin(), group_col],
        })
    return pd.DataFrame(rows)


def print_table(df, title):
    print(f'\n{"=" * 70}')
    print(f'  {title}')
    print(f'{"=" * 70}')
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


def main():
    # ── device ──────────────────────────────────────────────────────────────
    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    # ── model ───────────────────────────────────────────────────────────────
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    feature_matrix = sp.load_npz(
        os.path.join(args.data_path, args.dataset, 'feature.npz')
    )
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = CDD(data_generator.n_users,
                data_generator.n_items,
                norm_adj,
                feature_matrix,
                args)

    # Auto-detect checkpoint: prefer CLI --checkpoint, else latest .pkl
    ckpt_path = getattr(args, 'checkpoint', "model/449.pkl")
    if ckpt_path is None or not os.path.exists(ckpt_path):
        pkls = sorted(glob.glob(os.path.join(args.weights_path, '*.pkl')))
        if not pkls:
            sys.exit(f'[fairness_eval] No checkpoint found in {args.weights_path}')
        ckpt_path = pkls[-1]

    print(f'[fairness_eval] Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # ── demographics ────────────────────────────────────────────────────────
    feat_path = os.path.join(args.data_path, args.dataset, 'patient_fix_features.csv')
    if not os.path.exists(feat_path):
        sys.exit(f'[fairness_eval] Feature file not found: {feat_path}')

    demo_df = load_demographics(feat_path, data_generator.n_users)
    test_user_set = set(data_generator.test_set.keys())

    metric_cols = (
        [f'precision@{k}' for k in Ks] +
        [f'recall@{k}'    for k in Ks] +
        [f'ndcg@{k}'      for k in Ks] +
        [f'mrr@{k}'       for k in Ks] +
        [f'hit@{k}'       for k in Ks]
    )

    all_tables   = {}
    all_disparity = {}

    for axis, col in [('Age', 'age_group'), ('Sex', 'sex'), ('Race', 'race')]:
        print(f'\n[fairness_eval] Evaluating subgroups: {axis} ...')
        df = eval_subgroups(model, demo_df, col, test_user_set)
        disp = disparity_summary(df, col, metric_cols)

        print_table(df,   f'Per-{axis} Metrics')
        print_table(disp, f'{axis} Disparity Summary')

        all_tables[axis]    = df
        all_disparity[axis] = disp

    # ── save ────────────────────────────────────────────────────────────────
    os.makedirs(args.weights_path, exist_ok=True)
    out_prefix = os.path.join(args.weights_path, 'fairness')

    with pd.ExcelWriter(out_prefix + '.xlsx') as writer:
        for axis, df in all_tables.items():
            df.to_excel(writer, sheet_name=f'{axis}_metrics', index=False)
        for axis, df in all_disparity.items():
            df.to_excel(writer, sheet_name=f'{axis}_disparity', index=False)

    # Also write flat TSVs
    pd.concat(all_tables.values(),    keys=all_tables.keys()   ).to_csv(out_prefix + '_metrics.tsv',   sep='\t')
    pd.concat(all_disparity.values(), keys=all_disparity.keys()).to_csv(out_prefix + '_disparity.tsv', sep='\t')

    print(f'\n[fairness_eval] Results saved to {out_prefix}{{.xlsx,_metrics.tsv,_disparity.tsv}}')


if __name__ == '__main__':
    main()
