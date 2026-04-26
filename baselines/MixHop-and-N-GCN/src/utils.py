"""Data reading tools."""

import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    # t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    return graph

def feature_reader(path):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param path: Path to the JSON file.
    :return out_features: Dict with index and value tensor.
    """
    features = json.load(open(path))
    index_1 = [int(k) for k, v in features.items() for fet in v]
    index_2 = [int(fet) for k, v in features.items() for fet in v]
    values = [1.0]*len(index_1)
    nodes = [int(k) for k, v in features.items()]
    node_count = max(nodes)+1
    feature_count = max(index_2)+1
    features = sparse.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T)
    out_features["values"] = torch.FloatTensor(features.data)
    out_features["dimensions"] = features.shape
    return out_features

def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = torch.LongTensor(np.array(pd.read_csv(path)["target"]))
    return target

def create_adjacency_matrix(graph):
    """
    Creating a sparse adjacency matrix.
    :param graph: NetworkX object.
    :return A: Adjacency matrix.
    """
    index_1 = [edge[0] for edge in graph.edges()] + [edge[1] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()] + [edge[0] for edge in graph.edges()]
    values = [1 for index in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((values, (index_1, index_2)),
                          shape=(node_count, node_count),
                          dtype=np.float32)
    return A

def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(graph):
    """
    Creating a propagator matrix.
    :param graph: NetworkX graph.
    :return propagator: Dictionary of matrix indices and values.
    """
    A = create_adjacency_matrix(graph)
    I = sparse.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    propagator["indices"] = torch.LongTensor(ind.T)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
    return propagator


def load_mimic_propagator(norm_adj_npz_path):
    """
    Load precomputed normalized bipartite adjacency (same as NGCF/CDD).
    :param norm_adj_npz_path: Path to s_norm_adj_mat.npz
    :return propagator dict for MixHop layers
    """
    A = sparse.load_npz(norm_adj_npz_path).tocoo()
    ind = np.concatenate([A.row.reshape(-1, 1), A.col.reshape(-1, 1)], axis=1)
    propagator = dict()
    propagator["indices"] = torch.LongTensor(ind.T)
    propagator["values"] = torch.FloatTensor(A.data.astype(np.float32))
    return propagator


def load_mimic_train_test(train_path, test_path):
    """
    Parse train.txt / test.txt: each line: user_id item1 item2 ...
    :return train_items dict, test_set dict, n_users, n_items
    """
    train_items = {}
    n_users, n_items = 0, 0
    with open(train_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            uid = int(parts[0])
            items = [int(float(x)) for x in parts[1:]]
            train_items[uid] = items
            n_users = max(n_users, uid)
            if items:
                n_items = max(n_items, max(items))
    test_set = {}
    with open(test_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            uid = int(parts[0])
            items = [int(float(x)) for x in parts[1:]]
            test_set[uid] = items
            n_users = max(n_users, uid)
            if items:
                n_items = max(n_items, max(items))
    n_users += 1
    n_items += 1
    return train_items, test_set, n_users, n_items


def load_mimic_features(patient_csv_path, n_users, n_items):
    """
    User rows: demographics one-hot (sparse). Item rows: one-hot item id in block after user feats.
    :return feature dict compatible with SparseNGCNLayer (indices/values/dimensions)
    """
    df = pd.read_csv(patient_csv_path)
    if df.shape[0] != n_users:
        raise ValueError(
            "patient_fix_features rows (%d) must match n_users (%d)"
            % (df.shape[0], n_users)
        )
    n_user_feats = df.shape[1]
    feat_dim = n_user_feats + n_items
    rows, cols, vals = [], [], []
    for u in range(n_users):
        row = df.iloc[u]
        for j, colname in enumerate(df.columns):
            v = row[colname]
            if isinstance(v, str):
                active = v.lower() in ("true", "1", "yes")
            else:
                active = bool(v)
            if active:
                rows.append(u)
                cols.append(j)
                vals.append(1.0)
    offset = n_user_feats
    for local_i, global_row in enumerate(range(n_users, n_users + n_items)):
        rows.append(global_row)
        cols.append(offset + local_i)
        vals.append(1.0)
    node_count = n_users + n_items
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(node_count, feat_dim), dtype=np.float32)
    ind = np.concatenate([mat.row.reshape(-1, 1), mat.col.reshape(-1, 1)], axis=1)
    out_features = dict()
    out_features["indices"] = torch.LongTensor(ind.T)
    out_features["values"] = torch.FloatTensor(mat.data)
    out_features["dimensions"] = mat.shape
    return out_features


def load_mimic_graph_from_edges(edge_csv_path):
    """
    Read bipartite MIMIC edges where id_1=user_id and id_2=item_id (raw),
    then map to a unified node space: user u -> u, item i -> n_users + i.
    :return: graph, n_users, n_items
    """
    df = pd.read_csv(edge_csv_path)
    users = df["id_1"].astype(int).to_numpy()
    items = df["id_2"].astype(int).to_numpy()
    n_users = int(users.max()) + 1
    n_items = int(items.max()) + 1
    mapped_edges = [(int(u), int(n_users + i)) for u, i in zip(users, items)]
    graph = nx.Graph()
    graph.add_edges_from(mapped_edges)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    return graph, n_users, n_items


def load_mimic_node_features_from_json(feature_json_path, n_users, n_items):
    """
    Build full-node sparse feature matrix from patient-only feature json.
    User nodes keep demographic sparse features; item nodes get one-hot block.
    """
    features = json.load(open(feature_json_path, "r", encoding="utf-8"))
    n_user_feats = 0
    for v in features.values():
        if v:
            n_user_feats = max(n_user_feats, max(int(x) for x in v) + 1)
    feat_dim = n_user_feats + n_items
    rows, cols, vals = [], [], []
    for u in range(n_users):
        for fidx in features.get(str(u), []):
            j = int(fidx)
            if 0 <= j < n_user_feats:
                rows.append(u)
                cols.append(j)
                vals.append(1.0)
    offset = n_user_feats
    for i in range(n_items):
        rows.append(n_users + i)
        cols.append(offset + i)
        vals.append(1.0)
    mat = sparse.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_users + n_items, feat_dim),
        dtype=np.float32,
    )
    ind = np.concatenate([mat.row.reshape(-1, 1), mat.col.reshape(-1, 1)], axis=1)
    out_features = dict()
    out_features["indices"] = torch.LongTensor(ind.T)
    out_features["values"] = torch.FloatTensor(mat.data)
    out_features["dimensions"] = mat.shape
    return out_features


def load_mimic_features_from_npz(feature_npz_path, n_users, n_items):
    """
    Load user features from feature.npz (shape: n_users x n_user_feats).
    Extend to the full bipartite graph: user rows keep their sparse features,
    item rows get a one-hot block appended after the user feature columns.
    :param feature_npz_path: Path to feature.npz
    :param n_users: number of user nodes
    :param n_items: number of item nodes
    :return: feature dict with indices/values/dimensions
    """
    feat_mat = sparse.load_npz(feature_npz_path).tocoo().astype(np.float32)
    n_user_feats = feat_mat.shape[1]
    feat_dim = n_user_feats + n_items

    rows = feat_mat.row.tolist()
    cols = feat_mat.col.tolist()
    vals = feat_mat.data.tolist()

    for i in range(n_items):
        rows.append(n_users + i)
        cols.append(n_user_feats + i)
        vals.append(1.0)

    mat = sparse.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_users + n_items, feat_dim),
        dtype=np.float32,
    )
    ind = np.concatenate([mat.row.reshape(-1, 1), mat.col.reshape(-1, 1)], axis=1)
    return {
        "indices": torch.LongTensor(ind.T),
        "values": torch.FloatTensor(mat.data),
        "dimensions": mat.shape,
    }


def load_mimic_target_full(target_csv_path, n_users, n_items):
    """
    Read patient-level targets and extend to full node target vector.
    Item node (n_users+i) target is class i.
    """
    df = pd.read_csv(target_csv_path)
    user_target = np.zeros(n_users, dtype=np.int64)
    for _, row in df.iterrows():
        u = int(row["id"])
        if 0 <= u < n_users:
            user_target[u] = int(row["target"])
    item_target = np.arange(n_items, dtype=np.int64)
    full_target = np.concatenate([user_target, item_target], axis=0)
    return torch.LongTensor(full_target)
