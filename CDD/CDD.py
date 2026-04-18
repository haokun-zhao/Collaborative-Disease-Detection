'''
Created on March 24, 2023

@author: Haokun Zhao (18392891510@163.com)
'''

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class CDD(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, feature_matrix, args):#shortest_path_file
        super(CDD, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.emb_size = args.embed_size             # 64
        self.batch_size = args.batch_size           # 1024
        self.node_dropout = args.node_dropout[0]    # 0.1
        self.mess_dropout = args.mess_dropout       # [0.1, 0.1, 0.1]

        self.norm_adj = norm_adj
        # self.shortest_path_file = shortest_path_file

        self.layers = eval(args.layer_size) # [64,64,64]
        self.decay = eval(args.regs)[0]     # lambda = 1e-5
        self.use_demographics = bool(getattr(args, 'use_demographics', 1))
        self.hop_mixing = getattr(args, 'hop_mixing', 'adaptive')
        self.aggregator_mode = getattr(args, 'aggregator_mode', 'sum_bi')
        self.inter_layer_agg = getattr(args, 'inter_layer_agg', 'concat')

        """
        *********************************************************
        Load the feature matrix
        """
        self.feature = self._convert_sp_mat_to_sp_tensor(feature_matrix).to(self.device)
        self.feature.requires_grad = False
        # print(self.feature.shape)   torch.Size([61191, 44])
        # Cache dense feature matrix once so forward() doesn't re-allocate it every call.
        self.feature_dense_cache = self.feature.to_dense()
        if not self.use_demographics:
            self.feature_dense_cache.zero_()

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict1, self.weight_dict2 = self.init_weight()

        """
        *********************************************************
        Load the shortest path graph
        """
        # self.sparse_shortest_path_adj = self._convert_sp_mat_to_sp_tensor(self.shortest_path_file).to(self.device)

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

        # multi-order aggregation settings (1..K), K=3 to match previous 1..3
        self.K = int(getattr(args, 'max_hop', 3))
        self.alpha_params = nn.ParameterList()
        for _ in range(len(self.layers)):
            alpha = nn.Parameter(torch.zeros(self.K))  # Only K parameters for 1..K orders
            self.alpha_params.append(alpha)

    def init_weight(self):
        # xavier init
        # initialize the weights of each layer, maintain a stable variance of activations and gradients across different layers
        n = 3
        self.alpha = nn.Parameter(torch.randn(n))

        # Use softmax to ensure non-negative weights that sum to 1
        self.weights = F.softmax(self.alpha, dim=0)
        
        initializer = nn.init.xavier_uniform_   

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb1': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size))),
            'item_emb2': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.feature.shape[1])))
        })

        weight_dict1 = nn.ParameterDict({
            f'w{i+1}': nn.Parameter(self.weights[i].unsqueeze(0)) for i in range(n)
        })
        weight_dict2 = nn.ParameterDict()

        # layers = [self.emb_size] + self.layers  # layers = [64,64,64,64,64,64,64,64]
        feature_dim = self.feature.shape[1]   # 44
        input_dim   = self.emb_size + feature_dim  # 64 + 44 = 108
        layers      = [input_dim] + self.layers
        for k in range(len(self.layers)): # k = 0,1,2,3,4,5,6 update new key-item pairs
            # nn.Parameter() is trainable
            weight_dict2.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict2.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict2.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict2.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict1, weight_dict2

    def get_beta_weights(self, layer_idx, num_orders):
        """
        β_i^(l) = softmax(α^(l))
        """
        alpha_l = self.alpha_params[layer_idx]
        if self.hop_mixing == 'uniform':
            beta = torch.ones(num_orders, device=alpha_l.device) / float(num_orders)
        else:
            beta = torch.softmax(alpha_l[:num_orders], dim=0)
        return beta

    def multi_order_aggregation(self, A_hat, embeddings, layer_idx):
        """
        Aggregation over 1..K orders with learnable weights per layer
        Z = Σ_{i=1..K} β_i^(l) A^i E
        """
        num_orders = self.K
        beta_weights = self.get_beta_weights(layer_idx, num_orders)

        # Start with 1st order: A^1 * E
        A1_result = torch.sparse.mm(A_hat, embeddings)
        weighted_sum = beta_weights[0] * A1_result

        # Higher orders: A^i = A * A^(i-1)
        current_prop = A1_result
        for i in range(1, num_orders):
            current_prop = torch.sparse.mm(A_hat, current_prop)
            weighted_sum = weighted_sum + beta_weights[i] * current_prop
            # Clear intermediate results to save memory
            if i < num_orders - 1:
                torch.cuda.empty_cache()

        return weighted_sum

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()                             # COO: coordinate format of sparse matrix
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate    # 0.9
        random_tensor += torch.rand(noise_shape).to(x.device) #[0,1)
        # Each element of the mask is set to True with probability (1-rate) and 
        # False with probability (rate).
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        # Only the elements corresponding to the True entries in the dropout mask are retained, while others are set to zero.
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        # This tensor has the same shape as x and contains the dropout-applied values.
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        # Scale Dropout: This ensures that the expected value of the output remains the same during training and inference.
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    '''
    Need to be changed by a 4-layer NN
    '''
    def rating(self, u_g_embeddings, pos_i_g_embeddings): 
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    # need to be changed
    def get_initial_embeddings(self):
        """
        Ego embeddings before graph propagation: [n_user+n_item, emb_size + feature_dim].
        Used for influence attribution (gradients w.r.t. initial node inputs).
        """
        temp_embeddings = torch.cat([self.feature_dense_cache, self.embedding_dict['item_emb2']], 0)
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb1']], 0)
        ego_embeddings = torch.cat([ego_embeddings, temp_embeddings], dim=1)
        return ego_embeddings

    def _propagate_from_ego(self, A_hat, ego_embeddings):
        """Run L layers of message passing starting from ego_embeddings (differentiable)."""
        all_embeddings = [ego_embeddings]
        for k in range(len(self.layers)):
            side_embeddings = self.multi_order_aggregation(A_hat, ego_embeddings, k)
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict2['W_gc_%d' % k]) \
                                             + self.weight_dict2['b_gc_%d' % k]
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict2['W_bi_%d' % k]) \
                                            + self.weight_dict2['b_bi_%d' % k]
            if self.aggregator_mode == 'sum':
                ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings)
            elif self.aggregator_mode == 'bi':
                ego_embeddings = bi_embeddings
            else:
                ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings) + bi_embeddings
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]
            self._last_norm_embeddings = norm_embeddings

        if self.inter_layer_agg == 'mean':
            all_embeddings = torch.stack(all_embeddings[1:], dim=0).mean(dim=0)
        elif self.inter_layer_agg == 'last':
            all_embeddings = all_embeddings[-1]
        else:
            all_embeddings = torch.cat(all_embeddings, 1)
        return all_embeddings

    def forward_from_init(self, z_init_external, drop_flag=False):
        """
        Same propagation as forward(), but initial ego embeddings come from z_init_external
        (requires_grad=True) instead of nn.ParameterDict, so autograd flows into z_init.
        z_init_external: [n_user + n_item, emb_size + feature_dim]
        Returns: [n_user + n_item, final_dim] full node embeddings after propagation.
        """
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        return self._propagate_from_ego(A_hat, z_init_external)

    def forward_full_graph_embeddings(self, drop_flag=False):
        """All user and item final embeddings (for inference / predict_all)."""
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        ego_embeddings = self.get_initial_embeddings()
        all_embeddings = self._propagate_from_ego(A_hat, ego_embeddings)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]
        return u_g_embeddings, i_g_embeddings

    def predict_all_scores(self, patient_id, train_item_indices=None, drop_flag=False):
        """
        Dot-product scores for one patient vs all items: shape [n_item].
        Training positives can be masked to -inf for ranking.
        """
        u_emb, i_emb = self.forward_full_graph_embeddings(drop_flag=drop_flag)
        z_p = u_emb[int(patient_id)]
        scores = torch.matmul(i_emb, z_p)
        if train_item_indices is not None and len(train_item_indices) > 0:
            scores = scores.clone()
            scores[torch.tensor(train_item_indices, dtype=torch.long, device=scores.device)] = float("-inf")
        return scores

    def forward(self, users, pos_items, neg_items, drop_flag=True):
        # _nnz(): get the number of non-zero elements in the sparse normalized adjacency matrix 

        # Sparse dropout randomly sets elements of the sparse adjacency matrix to zero to prevent overfitting, 
        # and normalization ensures that the sum of the weights connecting each node to its neighbors is consistent.

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,  # 0.1
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = self.get_initial_embeddings()

        all_embeddings = self._propagate_from_ego(A_hat, ego_embeddings)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]
        # Use .clone().detach() so the large all_embeddings / norm_embeddings tensors
        # can be freed when forward() returns instead of being kept alive via a view.
        self.final_item_embeddings = all_embeddings[self.n_user:, :].clone().detach()
        self.final_user_embeddings = all_embeddings[:self.n_user, :].clone().detach()
        ln = self._last_norm_embeddings
        self.last_layer_item_embeddings = ln[self.n_user:, :].clone().detach()
        self.last_layer_user_embeddings = ln[:self.n_user, :].clone().detach()
        """
        *********************************************************
        look up.
        """
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
