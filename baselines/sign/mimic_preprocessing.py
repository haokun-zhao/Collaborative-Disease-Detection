# Copyright 2020 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
MIMIC-IV preprocessing for the SIGN baseline.

Loads the precomputed normalized adjacency matrix and patient features,
propagates features K times (X, AX, A^2X, ...) and saves the result as a
.pt file that sign_training.py --mimic can consume.

Node layout (mirrors CDD / MixHop):
  rows 0 .. n_users-1            : patient nodes  (44-dim demographic features)
  rows n_users .. n_users+n_items-1 : disease nodes (one-hot, n_items-dim block)

Full feature matrix X has shape (n_users + n_items, n_user_feats + n_items).
Each propagation step computes X <- A_hat @ X (sparse-dense multiply).
Only user rows (0..n_users-1) are stored per step to keep file size manageable.

Usage:
    cd sign/
    python mimic_preprocessing.py                       # default: 3 hops
    python mimic_preprocessing.py --num_propagations 2  # fewer hops

Output: mimic_sign.pt  (or the file given by --output)
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import trange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_train_test(train_path, test_path):
    train_items, test_set = {}, {}
    n_users, n_items = 0, 0
    for path, store in [(train_path, train_items), (test_path, test_set)]:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                uid = int(parts[0])
                items = [int(float(x)) for x in parts[1:]]
                store[uid] = items
                n_users = max(n_users, uid)
                if items:
                    n_items = max(n_items, max(items))
    return train_items, test_set, n_users + 1, n_items + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MIMIC-IV preprocessing for SIGN')
    parser.add_argument('--data_dir', type=str,
                        default=str(Path(__file__).resolve().parents[1] / "Data" / "mimicIV"),
                        help='Path to MIMIC-IV data directory')
    parser.add_argument('--num_propagations', type=int, default=3,
                        help='Number of propagation hops K (produces K+1 feature matrices)')
    parser.add_argument('--output', type=str, default='mimic_sign.pt',
                        help='Output .pt file for sign_training.py')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ------------------------------------------------------------------
    # 1. Train / test splits
    # ------------------------------------------------------------------
    print("Loading train/test splits ...")
    train_items, test_set, n_users, n_items = load_train_test(
        str(data_dir / "train2.txt"), str(data_dir / "test2.txt")
    )
    print(f"  n_users={n_users}  n_items={n_items}")

    # ------------------------------------------------------------------
    # 2. Normalized adjacency (precomputed, shared with CDD / MixHop)
    # ------------------------------------------------------------------
    print("Loading normalized adjacency matrix ...")
    norm_adj = sp.load_npz(str(data_dir / "s_norm_adj_mat2.npz")).tocsr().astype(np.float32)
    print(f"  shape={norm_adj.shape}  nnz={norm_adj.nnz}")

    # ------------------------------------------------------------------
    # 3. Build full-graph feature matrix X  (n_total, feat_dim)
    #    user rows : 44-dim demographic features
    #    item rows : n_items-dim one-hot block
    # ------------------------------------------------------------------
    print("Building full-graph feature matrix ...")
    feat_mat = sp.load_npz(str(data_dir / "feature.npz")).astype(np.float32)
    n_user_feats = feat_mat.shape[1]       # 44
    feat_dim = n_user_feats + n_items      # 44 + 2000 = 2044
    n_total = n_users + n_items            # 63191

    # user block : [feat_mat | 0_{n_users x n_items}]
    user_block = sp.hstack(
        [feat_mat.tocsr(),
         sp.csr_matrix((n_users, n_items), dtype=np.float32)],
        format="csr",
    )
    # item block : [0_{n_items x n_user_feats} | I_{n_items}]
    item_block = sp.hstack(
        [sp.csr_matrix((n_items, n_user_feats), dtype=np.float32),
         sp.eye(n_items, dtype=np.float32).tocsr()],
        format="csr",
    )
    X = sp.vstack([user_block, item_block], format="csr")   # (n_total, feat_dim)
    print(f"  X shape={X.shape}  nnz={X.nnz}")

    # ------------------------------------------------------------------
    # 4. Propagate: compute X, AX, A^2X, ...  (keep only user rows)
    # ------------------------------------------------------------------
    print(f"Computing {args.num_propagations} propagation(s) ...")
    op_embedding = []

    # hop 0 : raw features for user nodes  (stored as float16 to halve RAM)
    x_user = torch.from_numpy(X[:n_users].toarray()).half()
    op_embedding.append(x_user)
    print(f"  hop 0  user-feature shape: {x_user.shape}  dtype={x_user.dtype}")
    del x_user

    x_prop = X.copy()
    for k in trange(args.num_propagations, desc="Propagating"):
        x_prop = norm_adj @ x_prop          # sparse @ sparse → sparse (csr)
        # Extract user rows, densify, store as float16
        x_user_k = torch.from_numpy(x_prop[:n_users].toarray()).half()
        op_embedding.append(x_user_k)
        print(f"  hop {k+1}  user-feature shape: {x_user_k.shape}  dtype={x_user_k.dtype}")
        del x_user_k
    del x_prop

    # ------------------------------------------------------------------
    # 5. Train / validation split on user indices (90 / 10)
    # NOTE: dense multi-hot labels are NOT stored; train_items carries the
    # same information and costs ~0 MB vs ~490 MB for the dense matrix.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(np.arange(n_users))
    n_val = max(1, int(0.1 * n_users))
    val_idx   = torch.from_numpy(shuffled[:n_val].copy())
    train_idx = torch.from_numpy(shuffled[n_val:].copy())

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    op_dict = {
        "op_embedding": op_embedding,           # list[(n_users, feat_dim)] float16 tensors
        "n_users":      n_users,
        "n_items":      n_items,
        "split_idx":    {"train": train_idx, "valid": val_idx},
        "train_items":  train_items,
        "test_set":     test_set,
    }
    torch.save(op_dict, args.output)
    total_feat_dim = feat_dim * (args.num_propagations + 1)
    hop_mb = (n_users * feat_dim * 2) / 1024**2          # float16
    total_mb = hop_mb * (args.num_propagations + 1)
    print(f"\nSaved to '{args.output}'")
    print(f"  Per-hop feature dim    : {feat_dim}  ({hop_mb:.0f} MB each, float16)")
    print(f"  Total input dim (concat): {total_feat_dim}  ({total_mb:.0f} MB, float16)")
    print(f"  Training users : {len(train_idx)}   Validation users : {len(val_idx)}")
    print(f"  Labels omitted (use train_items dict; saves ~{n_users*n_items*4/1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
