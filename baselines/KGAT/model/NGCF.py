import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class NGCF(nn.Module):
    """
    Neural Graph Collaborative Filtering (Wang et al., SIGIR 2019).
    https://dl.acm.org/doi/10.1145/3331184.3331267
    """

    def __init__(self, args, n_users, n_items, norm_adj,
                 user_pre_embed=None, item_pre_embed=None):
        super(NGCF, self).__init__()

        self.n_users      = n_users
        self.n_items      = n_items
        self.embed_dim    = args.embed_dim
        self.layers       = eval(args.layer_size)   # e.g. [64, 64, 64]
        self.n_layers     = len(self.layers)
        self.mess_dropout = eval(args.mess_dropout) # e.g. [0.1, 0.1, 0.1]
        self.l2loss_lambda = args.l2loss_lambda

        # Pre-computed sparse normalised adjacency  (n_users+n_items) × (n_users+n_items)
        self.norm_adj = self._sp_to_tensor(norm_adj)

        # ── initial embeddings ───────────────────────────────────────────
        self.user_embed = nn.Embedding(n_users, self.embed_dim)
        self.item_embed = nn.Embedding(n_items, self.embed_dim)

        if user_pre_embed is not None:
            self.user_embed.weight = nn.Parameter(user_pre_embed)
        else:
            nn.init.xavier_uniform_(self.user_embed.weight)

        if item_pre_embed is not None:
            self.item_embed.weight = nn.Parameter(item_pre_embed)
        else:
            nn.init.xavier_uniform_(self.item_embed.weight)

        # ── propagation weight matrices W_1^(l) and W_2^(l) ─────────────
        self.W1 = nn.ModuleList()
        self.W2 = nn.ModuleList()
        in_dim = self.embed_dim
        for out_dim in self.layers:
            self.W1.append(nn.Linear(in_dim, out_dim, bias=False))
            self.W2.append(nn.Linear(in_dim, out_dim, bias=False))
            in_dim = out_dim

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.drop_layers = nn.ModuleList(
            [nn.Dropout(p=d) for d in self.mess_dropout]
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _sp_to_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        idx = torch.LongTensor(np.vstack([coo.row, coo.col]))
        val = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(idx, val, coo.shape)

    # ── graph propagation ────────────────────────────────────────────────

    def get_propagated_embeddings(self):
        """Full-graph NGCF propagation; returns (u_emb, i_emb) for batched scoring (eval)."""
        return self._propagate()

    def _propagate(self):
        """Full-graph NGCF propagation; returns (u_emb, i_emb) after L layers."""
        device = self.user_embed.weight.device
        A = self.norm_adj.to(device)

        E = torch.cat([self.user_embed.weight, self.item_embed.weight], dim=0)
        # (n_users + n_items, embed_dim)

        layer_embs = [E]
        for l in range(self.n_layers):
            E_gc = torch.sparse.mm(A, E)          # neighbourhood aggregation
            side1 = self.W1[l](E + E_gc)          # interaction & self propagation
            side2 = self.W2[l](E_gc * E)          # element-wise interaction
            E = self.leakyrelu(side1 + side2)
            if self.training:
                E = self.drop_layers[l](E)
            E = nn.functional.normalize(E, p=2, dim=1)
            layer_embs.append(E)

        # Concatenate all-layer embeddings (0 … L)
        E_all = torch.cat(layer_embs, dim=1)      # (n_nodes, embed_dim * (L+1))
        u_emb = E_all[: self.n_users]
        i_emb = E_all[self.n_users :]
        return u_emb, i_emb

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, user_ids, item_ids, is_train=True):
        if is_train:
            pos_ids, neg_ids = item_ids
            return self._bpr_loss(user_ids, pos_ids, neg_ids)
        else:
            return self._calc_score(user_ids, item_ids)

    def _calc_score(self, user_ids, item_ids):
        u_emb, i_emb = self._propagate()
        u = u_emb[user_ids]          # (batch_users, D)
        i = i_emb[item_ids]          # (n_items,    D)
        return torch.matmul(u, i.t())  # (batch_users, n_items)

    def _bpr_loss(self, user_ids, pos_ids, neg_ids):
        u_emb, i_emb = self._propagate()
        u   = u_emb[user_ids]
        pos = i_emb[pos_ids]
        neg = i_emb[neg_ids]

        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # L2 regularisation on initial (layer-0) embeddings only
        reg = (_L2_loss_mean(self.user_embed(user_ids))
               + _L2_loss_mean(self.item_embed(pos_ids))
               + _L2_loss_mean(self.item_embed(neg_ids)))

        return bpr_loss + self.l2loss_lambda * reg
