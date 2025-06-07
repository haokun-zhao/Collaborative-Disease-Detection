'''
Created on March 24, 2023

@author: Haokun Zhao (18392891510@163.com)
'''

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class CDD(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, feature_matrix, args):#shortest_path_file
        super(CDD, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size             # 64
        self.batch_size = args.batch_size           # 1024
        self.node_dropout = args.node_dropout[0]    # 0.1
        self.mess_dropout = args.mess_dropout       # [0.1, 0.1, 0.1]

        self.norm_adj = norm_adj
        # self.shortest_path_file = shortest_path_file

        self.layers = eval(args.layer_size) # [64,64,64,64,64,64]
        self.decay = eval(args.regs)[0]     # lambda = 1e-5

        """
        *********************************************************
        Load the feature matrix
        """
        self.feature = self._convert_sp_mat_to_sp_tensor(feature_matrix).to(self.device)
        self.feature.requires_grad = False
        # print(self.feature.shape)   torch.Size([61191, 44])

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
        self.adj_matrices = self.init_adj_matrices()

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

        layers = [self.emb_size] + self.layers  # layers = [64,64,64,64,64,64,64,64]
        for k in range(len(self.layers)): # k = 0,1,2,3,4,5,6 update new key-item pairs
            # nn.Parameter() is trainable
            weight_dict2.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict2.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict2.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict2.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict1, weight_dict2

    def init_adj_matrices(self):
        # Initialize A_hat_1, A_hat_2, A_hat_3 to A_hat_7
        adj_matrices = []#[self.sparse_shortest_path_adj]
        A_hat = self.sparse_norm_adj

        for i in range(1, 4):
            A_hat = A_hat ** i
            adj_matrices.append(A_hat)

        return adj_matrices

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
        # device = u_g_embeddings.device
        # u_batch_size = u_g_embeddings.size(0)
        # i_batch_size = pos_i_g_embeddings.size(0)
        # # Perform matrix multiplication
        # interactions = torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

        # # Initialize rating function network
        # rating_net = RatingNetwork(u_batch_size, i_batch_size).to(device)

        # # Flatten the interactions matrix to feed each value to the neural network
        # interactions_flat = interactions.view(-1)
        # interactions_flat = torch.tensor(interactions_flat)

        # # Feed each value in the interactions matrix to the neural network
        # # ratings = []
        # # for val in interactions_flat:
        # #     val_tensor = torch.tensor([[val]])  # Create a tensor of shape (1, 1) for each value
        # rating = rating_net(interactions_flat)
        # # ratings.append(rating.item())  # Append the rating value to the list

        # # Reshape the ratings list to match the shape of the interactions matrix
        # ratings = torch.tensor(rating).view(u_batch_size, i_batch_size)

        # return ratings
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    # need to be changed
    def forward(self, users, pos_items, neg_items, drop_flag=True):
        # _nnz(): get the number of non-zero elements in the sparse normalized adjacency matrix 

        # Sparse dropout randomly sets elements of the sparse adjacency matrix to zero to prevent overfitting, 
        # and normalization ensures that the sum of the weights connecting each node to its neighbors is consistent.

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,  # 0.1
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        temp_embeddings = torch.cat([self.feature.to_dense(), self.embedding_dict['item_emb2']], 0)  # 63191*44

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],         # 61191*64
                                    self.embedding_dict['item_emb1']], 0)    # 2000 *64

        ego_embeddings = torch.cat([ego_embeddings, temp_embeddings], dim = 1) # 63191*108

        all_embeddings = [ego_embeddings]   # E
        # print(self.weight_dict['w1'].shape) # torch.Size([1])
        # for layer in range(len(self.layers)):   # len(self.layers) = 3
        #     # temp_emb = []

        #     # for i in range(len(self.adj_matrices)): # len(self.adj_matrices) = 5
        #         # print(self.adj_matrices[i].shape)   # 63191*2000
        #         # temp_emb.append(torch.sparse.mm(self.adj_matrices[i], all_embeddings[-1]))  # 63191*108
        #     # temp_emb = [torch.sparse.mm(adj_matrix, all_embeddings[-1]) for adj_matrix in self.adj_matrices]
        #     adj1 = torch.sparse.mm(self.adj_matrices[0], ego_embeddings) * self.weight_dict['w1']
        #     adj2 = torch.sparse.mm(self.adj_matrices[1], ego_embeddings) * self.weight_dict['w2']
        #     adj3 = torch.sparse.mm(self.adj_matrices[2], ego_embeddings) * self.weight_dict['w3']
        #     # adj4 = torch.sparse.mm(self.adj_matrices[3], ego_embeddings) * self.weight_dict['w4']
        #     # adj5 = torch.sparse.mm(self.adj_matrices[4], ego_embeddings) * self.weight_dict['w5']

        #     # Sum the new embeddings
        #     ego_embeddings = adj1 + adj2 + adj3 #+ adj4 + adj5

        #     # new_embeddings = sum([self.weight_dict[f'w{i+1}'] * temp_emb[i] for i in range(len(temp_emb))])
        #     # new_embeddings = sum(self.weight_dict[f'w{i+1}'] * emb for i, emb in enumerate(temp_emb))
        #     # print(new_embeddings.shape) torch.Size([63191, 108])
        #     # new_embeddings = F.leaky_relu(torch.matmul(new_embeddings, self.weight_dict[f'w{layer+1}']))
        #     ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(adj1 + adj2 + adj3)
        #     norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
        #     all_embeddings += [norm_embeddings]

        for k in range(len(self.layers)): # k = 0,1,2
            # embeddings of neighboring nodes
            # (L+I)*E
            adj1 = torch.sparse.mm(self.adj_matrices[0], ego_embeddings) * self.weight_dict1['w1']
            adj2 = torch.sparse.mm(self.adj_matrices[1], ego_embeddings) * self.weight_dict1['w2']
            adj3 = torch.sparse.mm(self.adj_matrices[2], ego_embeddings) * self.weight_dict1['w3']
            # side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)   # A_hat * ego_embeddings
            side_embeddings = adj1 + adj2 + adj3
            # sum of transformed messages of neighbors
            # (L+I)*E*W_1 + b_1
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict2['W_gc_%d' % k]) \
                                             + self.weight_dict2['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product 
            # e_i·e_u
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            # W_2*(e_i·e_u)+b_2
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict2['W_bi_%d' % k]) \
                                            + self.weight_dict2['b_bi_%d' % k]

            # non-linear activation.
            # E^{(l)}
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            # concate embeddings of each layer
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        """
        *********************************************************
        look up.
        """
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

# class MultiLayerPerceptron(nn.Module):

#     def __init__(self, input_dim, embed_dims, dropout, output_layer=True, output_dim=1):
#         super().__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(nn.Dropout(p=dropout))
#             layers.append(nn.Linear(input_dim, embed_dim))
#             layers.append(nn.ReLU())
#             input_dim = embed_dim

#         if output_layer:
#             layers.append(nn.Linear(input_dim, output_dim))

#         self.mlp = nn.Sequential(*layers)


#     def forward(self, x):
#         """
#         :param x: Float tensor of size ``(batch_size, embed_dim)``
#         """
#         return self.mlp(x.float())

# class RatingNetwork(nn.Module):
#     def __init__(self, u_batch_size, i_batch_size):
#         super(RatingNetwork, self).__init__()
#         self.u_batch_size = u_batch_size
#         self.i_batch_size = i_batch_size
#         self.fc1 = nn.Linear(u_batch_size * i_batch_size, 20).cuda()
#         self.fc2 = nn.Linear(20, 20).cuda()
#         self.fc3 = nn.Linear(20, u_batch_size * i_batch_size).cuda()

#     def forward(self, x):
#         x = torch.sigmoid(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x

# class RatingNetwork(nn.Module):
#     def __init__(self, u_batch_size, i_batch_size):
#         super(RatingNetwork, self).__init__()
#         self.u_batch_size = u_batch_size
#         self.i_batch_size = i_batch_size
#         self.fc1 = nn.Linear(u_batch_size * i_batch_size, 20).cuda()
#         self.fc2 = nn.Linear(20, 20).cuda()
#         self.fc3 = nn.Linear(20, u_batch_size * i_batch_size).cuda()
#         self.relu = nn.LeakyReLU(negative_slope=0.01)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x