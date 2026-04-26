"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
He et al., SIGIR 2020 — adapted for collaborative disease detection.

Key idea: strip all feature transformation matrices and non-linear activations;
only keep the light graph convolution (symmetric normalisation) and sum/mean
layer aggregation.
"""

import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np


class LightGCN(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(LightGCN, self).__init__()
        self.n_user   = n_user
        self.n_item   = n_item
        self.device   = args.device
        self.emb_size = args.embed_size
        self.n_layers = len(eval(args.layer_size))   # re-use layer_size arg for depth
        self.decay    = eval(args.regs)[0]
        self.batch_size = args.batch_size

        # Learnable embeddings (no feature transform — pure LightGCN)
        self.user_emb = nn.Embedding(n_user, self.emb_size)
        self.item_emb = nn.Embedding(n_item, self.emb_size)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # Pre-compute the symmetric-normalised adjacency (sparse, on device)
        self.norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj).to(self.device)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(i, v, coo.shape)

    # ------------------------------------------------------------------ forward
    def forward(self, users, pos_items, neg_items, drop_flag=False):
        """
        Light graph convolution over all layers; final embedding = mean of all layers.
        Returns lookup embeddings for the given users / items.
        """
        # Stack initial embeddings [n_user+n_item, emb_size]
        E0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        all_layer_emb = [E0]
        E = E0
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.norm_adj, E)
            all_layer_emb.append(E)

        # Mean-pool across layers (including layer 0)
        E_final = torch.stack(all_layer_emb, dim=1).mean(dim=1)

        self.final_user_emb = E_final[:self.n_user].detach()
        self.final_item_emb = E_final[self.n_user:].detach()

        def _to_tensor(idx):
            if isinstance(idx, torch.Tensor):
                return idx.to(self.device)
            return torch.tensor(list(idx), dtype=torch.long, device=self.device)

        u_idx   = _to_tensor(users)
        pos_idx = _to_tensor(pos_items)

        u_emb   = E_final[u_idx, :]
        pos_emb = E_final[self.n_user + pos_idx, :]

        if len(neg_items) > 0:
            neg_idx = _to_tensor(neg_items)
            neg_emb = E_final[self.n_user + neg_idx, :]
        else:
            neg_emb = torch.zeros(0, self.emb_size, device=self.device)

        return u_emb, pos_emb, neg_emb

    def rating(self, u_emb, i_emb):
        """Inner-product rating for evaluation."""
        return torch.matmul(u_emb, i_emb.t())

    # ------------------------------------------------------------------ loss
    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        mf_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        # L2 regularisation on the mini-batch embeddings (standard LightGCN)
        reg_loss = (users.norm(2).pow(2)
                    + pos_items.norm(2).pow(2)
                    + neg_items.norm(2).pow(2)) / 2
        emb_loss = self.decay * reg_loss / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
