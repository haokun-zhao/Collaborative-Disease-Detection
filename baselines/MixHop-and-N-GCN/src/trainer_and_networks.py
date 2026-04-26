import gc
import time
import torch
import random
import numpy as np
from tqdm import trange
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from utils import create_propagator_matrix
from layers import SparseNGCNLayer, DenseNGCNLayer, ListModule
from mimic_metrics import rank_and_metrics

class NGCNNetwork(torch.nn.Module):
    """
    Higher Order Graph Convolutional Model.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """
    def __init__(self, args, feature_number, class_number):
        super(NGCNNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.order = len(self.args.layers_1)
        self.setup_layer_structure()

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional layers) and dense final.
        """
        self.main_layers = [SparseNGCNLayer(self.feature_number, self.args.layers_1[i-1], i, self.args.dropout) for i in range(1, self.order+1)]
        self.main_layers = ListModule(*self.main_layers)
        self.fully_connected = torch.nn.Linear(sum(self.args.layers_1), self.class_number)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """
        abstract_features = [self.main_layers[i](normalized_adjacency_matrix, features) for i in range(self.order)]
        abstract_features = torch.cat(abstract_features, dim=1)
        predictions = self.fully_connected(abstract_features)
        if getattr(self.args, "dataset", "cora") in ("mimic", "eicu"):
            return predictions
        return torch.nn.functional.log_softmax(predictions, dim=1)

class MixHopNetwork(torch.nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """
    def __init__(self, args, feature_number, class_number):
        super(MixHopNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.calculate_layer_sizes()
        self.setup_layer_structure()

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.args.layers_1)
        self.abstract_feature_number_2 = sum(self.args.layers_2)
        self.order_1 = len(self.args.layers_1)
        self.order_2 = len(self.args.layers_2)

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        self.upper_layers = [SparseNGCNLayer(self.feature_number, self.args.layers_1[i-1], i, self.args.dropout) for i in range(1, self.order_1+1)]
        self.upper_layers = ListModule(*self.upper_layers)
        self.bottom_layers = [DenseNGCNLayer(self.abstract_feature_number_1, self.args.layers_2[i-1], i, self.args.dropout) for i in range(1, self.order_2+1)]
        self.bottom_layers = ListModule(*self.bottom_layers)
        self.fully_connected = torch.nn.Linear(self.abstract_feature_number_2, self.class_number)

    def calculate_group_loss(self):
        """
        Calculating the column losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            upper_column_loss = torch.norm(self.upper_layers[i].weight_matrix, dim=0)
            loss_upper = torch.sum(upper_column_loss)
            weight_loss = weight_loss + self.args.lambd*loss_upper
        for i in range(self.order_2):
            bottom_column_loss = torch.norm(self.bottom_layers[i].weight_matrix, dim=0)
            loss_bottom = torch.sum(bottom_column_loss)
            weight_loss = weight_loss + self.args.lambd*loss_bottom
        return weight_loss

    def calculate_loss(self):
        """
        Calculating the losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            loss_upper = torch.norm(self.upper_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd*loss_upper
        for i in range(self.order_2):
            loss_bottom = torch.norm(self.bottom_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd*loss_bottom
        return weight_loss
            
    def forward(self, normalized_adjacency_matrix, features):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """
        abstract_features_1 = torch.cat([self.upper_layers[i](normalized_adjacency_matrix, features) for i in range(self.order_1)], dim=1)
        abstract_features_2 = torch.cat([self.bottom_layers[i](normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)], dim=1)
        predictions = self.fully_connected(abstract_features_2)
        if getattr(self.args, "dataset", "cora") in ("mimic", "eicu"):
            return predictions
        return torch.nn.functional.log_softmax(predictions, dim=1)

class Trainer(object):
    """
    Class for training the neural network.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :param features: Feature sparse matrix.
    :param target: Target vector.
    :param base_run: Loss calculation behavioural flag.
    """
    def __init__(self, args, graph, features, target, base_run):
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self.base_run = base_run
        if getattr(self.args, "dataset", "cora") in ("mimic", "eicu"):
            self.setup_mimic_features()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.setup_features()
            self.train_test_split()
        self.setup_model()

    def train_test_split(self):
        """
        Creating a train/test split.
        """
        random.seed(self.args.seed)
        # For converted MIMIC inputs, evaluate on user nodes only.
        mimic_user_count = getattr(self.args, "mimic_user_count", None)
        if mimic_user_count is not None:
            nodes = [node for node in range(int(mimic_user_count))]
        else:
            nodes = [node for node in range(self.ncount)]
        random.shuffle(nodes)
        self.train_nodes = torch.LongTensor(nodes[0:self.args.training_size])
        self.validation_nodes = torch.LongTensor(nodes[self.args.training_size:self.args.training_size+self.args.validation_size])
        self.test_nodes = torch.LongTensor(nodes[self.args.training_size+self.args.validation_size:])

    def setup_mimic_features(self):
        """
        MIMIC-IV bipartite graph: multi-label diseases per user, precomputed norm adjacency.
        """
        self.n_users = self.args.n_users
        self.n_items = self.args.n_items
        self.ncount = self.n_users + self.n_items
        self.feature_number = self.features["dimensions"][1]
        self.class_number = self.n_items
        self.propagation_matrix = self.args.propagation_matrix
        self.train_items = self.args.train_items
        self.test_set = self.args.test_set
        self.train_multi_hot = torch.zeros(self.n_users, self.n_items, dtype=torch.float32)
        for u, items in self.train_items.items():
            for it in items:
                self.train_multi_hot[u, it] = 1.0
        random.seed(self.args.seed)
        user_ids = list(range(self.n_users))
        random.shuffle(user_ids)
        n_val = max(1, int(0.1 * self.n_users))
        self.val_users = set(user_ids[:n_val])
        self.train_supervised_users = set(user_ids[n_val:])
        if getattr(self.args, "force_gpu", False):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.ncount > 20000:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                print(
                    "MIMIC graph has %d nodes; using CPU to reduce risk of GPU OOM. "
                    "Use --force-gpu to try CUDA (needs large VRAM).\n"
                    % self.ncount
                )
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_features(self):
        """
        Creating a feature matrix, target vector and propagation matrix.
        """
        self.ncount = self.features["dimensions"][0]
        self.feature_number = self.features["dimensions"][1]
        self.class_number = torch.max(self.target).item()+1
        self.propagation_matrix = create_propagator_matrix(self.graph)

    def setup_model(self):
        """
        Defining a PageRankNetwork.
        """
        if self.args.model == "mixhop":
            self.model = MixHopNetwork(self.args, self.feature_number, self.class_number)
        else:
            self.model = NGCNNetwork(self.args, self.feature_number, self.class_number)

    def _prop_features_to_device(self):
        pm = {
            "indices": self.propagation_matrix["indices"].to(self.device),
            "values": self.propagation_matrix["values"].to(self.device),
        }
        ft = {
            "indices": self.features["indices"].to(self.device),
            "values": self.features["values"].to(self.device),
            "dimensions": self.features["dimensions"],
        }
        return pm, ft

    def _aggregate_mimic_test_metrics(self, user_logits_np):
        ks = self.args.ks
        sum_auc  = 0.0
        sum_prec = np.zeros(len(ks))
        sum_rec  = np.zeros(len(ks))
        sum_ndcg = np.zeros(len(ks))
        sum_hit  = np.zeros(len(ks))
        sum_mrr  = np.zeros(len(ks))
        n_ev = 0
        for u, test_pos in self.test_set.items():
            train_pos = set(self.train_items.get(u, []))
            candidates = [i for i in range(self.n_items) if i not in train_pos]
            if not candidates or not test_pos:
                continue
            m = rank_and_metrics(set(test_pos), candidates, user_logits_np[u], ks)
            sum_auc  += m["auc"]
            sum_prec += np.array(m["precision"])
            sum_rec  += np.array(m["recall"])
            sum_ndcg += np.array(m["ndcg"])
            sum_hit  += np.array(m["hit_ratio"])
            sum_mrr  += np.array(m["mrr"])
            n_ev += 1
        return n_ev, sum_auc, sum_prec, sum_rec, sum_ndcg, sum_hit, sum_mrr

    def _print_mimic_ranking_aggregate(
        self, n_ev, sum_auc, sum_prec, sum_rec, sum_ndcg, sum_hit, sum_mrr, title
    ):
        ks = self.args.ks
        if n_ev == 0:
            print(title + " — no users evaluated.")
            return
        print(title + " — users: %d" % n_ev)
        print("  AUC: %.6f" % (sum_auc / n_ev))
        for j, k in enumerate(ks):
            print(
                "  K=%d  Precision: %.6f  Recall: %.6f  NDCG: %.6f  Hit: %.6f  MRR: %.6f"
                % (
                    k,
                    sum_prec[j] / n_ev,
                    sum_rec[j] / n_ev,
                    sum_ndcg[j] / n_ev,
                    sum_hit[j] / n_ev,
                    sum_mrr[j] / n_ev,
                )
            )

    def fit_mimic(self):
        """
        Train with multi-label BCE on user nodes; early stopping on held-out user val BCE.
        """
        self.model.to(self.device)
        self.train_multi_hot = self.train_multi_hot.to(self.device)
        pm, ft = self._prop_features_to_device()
        val_mask = torch.zeros(self.n_users, dtype=torch.bool, device=self.device)
        for u in self.val_users:
            val_mask[u] = True
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # Precompute BPR pair tensors (user, pos_item) for training
        _bpr_u, _bpr_p = [], []
        for u in sorted(self.train_supervised_users):
            for p in self.train_items.get(u, []):
                _bpr_u.append(u)
                _bpr_p.append(p)
        bpr_u_t = torch.LongTensor(_bpr_u).to(self.device)
        bpr_p_t = torch.LongTensor(_bpr_p).to(self.device)
        n_bpr_pairs = len(_bpr_u)
        del _bpr_u, _bpr_p
        best_val = float("inf")
        best_state = None
        no_improvement = 0
        self.epochs_trained = 0
        self.convergence_epoch = 0
        self.train_log = []          # [{epoch, train_loss, val_loss, epoch_time_s}, ...]
        metric_every = getattr(self.args, "metric_every", 5)
        epochs = trange(self.args.epochs, desc="MIMIC")
        for epoch_idx in epochs:
            _epoch_t0 = time.perf_counter()
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(pm, ft)
            user_logits = logits[: self.n_users]
            # BPR loss: sample negatives and optimize pairwise ranking
            neg_np = np.random.randint(0, self.n_items, size=n_bpr_pairs).astype(np.int64)
            bpr_n_t = torch.from_numpy(neg_np).to(self.device)
            pos_sc = user_logits[bpr_u_t, bpr_p_t]
            neg_sc = user_logits[bpr_u_t, bpr_n_t]
            loss = torch.nn.functional.softplus(neg_sc - pos_sc).mean()
            if self.args.model == "mixhop":
                loss = loss + self.model.calculate_loss()
            train_loss_val = float(loss.detach())
            loss.backward()
            optimizer.step()
            del logits, user_logits, pos_sc, neg_sc, bpr_n_t, loss
            gc.collect()
            self.model.eval()
            with torch.inference_mode():
                v_logits = self.model(pm, ft)[: self.n_users]
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    v_logits[val_mask], self.train_multi_hot[val_mask]
                ).item()
            del v_logits
            gc.collect()
            epoch_time_s = time.perf_counter() - _epoch_t0
            self.train_log.append({
                "epoch":        epoch_idx + 1,
                "train_loss":   round(train_loss_val, 6),
                "val_loss":     round(val_loss, 6),
                "epoch_time_s": round(epoch_time_s, 4),
            })
            print(
                "Epoch %d/%d  train_loss=%.6f  val_bce=%.6f  time=%.2fs"
                % (epoch_idx + 1, self.args.epochs, train_loss_val, val_loss, epoch_time_s)
            )
            epochs.set_description(
                "Ep %d train=%.4f val=%.4f"
                % (epoch_idx + 1, train_loss_val, val_loss)
            )
            if metric_every > 0 and (epoch_idx + 1) % metric_every == 0:
                with torch.inference_mode():
                    eval_logits = self.model(pm, ft)[: self.n_users].cpu().numpy()
                n_ev_mid, s_auc, s_prec, s_rec, s_ndcg, s_hit, s_mrr = \
                    self._aggregate_mimic_test_metrics(eval_logits)
                self._print_mimic_ranking_aggregate(
                    n_ev_mid, s_auc, s_prec, s_rec, s_ndcg, s_hit, s_mrr,
                    title="[Epoch %d] Test ranking metrics" % (epoch_idx + 1),
                )
                del eval_logits
                gc.collect()
            self.epochs_trained = epoch_idx + 1
            if val_loss < best_val:
                best_val = val_loss
                self.convergence_epoch = epoch_idx + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                no_improvement = 0
            else:
                no_improvement = no_improvement + 1
                if no_improvement == self.args.early_stopping:
                    epochs.close()
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        print("\nMIMIC training stopped. Best val BCE: " + str(round(best_val, 5)) + "\n")

    def evaluate_mimic_ranking(self, return_per_user=False):
        """
        AUC, Precision, Recall, NDCG, Hit Rate, MRR @ Ks.
        Prints results and returns a metrics dict for programmatic access.
        """
        self.model.eval()
        pm, ft = self._prop_features_to_device()
        _t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(pm, ft)
            user_logits = logits[: self.n_users].cpu().numpy()
        inference_time_s = time.perf_counter() - _t0
        del logits
        gc.collect()
        ks = self.args.ks
        per_user = []
        n_ev, sum_auc, sum_prec, sum_rec, sum_ndcg, sum_hit, sum_mrr = 0, 0.0, np.zeros(len(ks)), np.zeros(len(ks)), np.zeros(len(ks)), np.zeros(len(ks)), np.zeros(len(ks))
        for u, test_pos in self.test_set.items():
            train_pos = set(self.train_items.get(u, []))
            candidates = [i for i in range(self.n_items) if i not in train_pos]
            if not candidates or not test_pos:
                continue
            m = rank_and_metrics(set(test_pos), candidates, user_logits[u], ks)
            sum_auc += m["auc"]
            sum_prec += np.array(m["precision"])
            sum_rec += np.array(m["recall"])
            sum_ndcg += np.array(m["ndcg"])
            sum_hit += np.array(m["hit_ratio"])
            sum_mrr += np.array(m["mrr"])
            n_ev += 1
            if return_per_user:
                row = {"user_id": int(u), "auc": float(m["auc"]), "test_items_count": int(len(test_pos))}
                for j, k in enumerate(ks):
                    row[f"precision@{k}"] = float(m["precision"][j])
                    row[f"recall@{k}"] = float(m["recall"][j])
                    row[f"ndcg@{k}"] = float(m["ndcg"][j])
                    row[f"hit@{k}"] = float(m["hit_ratio"][j])
                    row[f"mrr@{k}"] = float(m["mrr"][j])
                per_user.append(row)
        if n_ev == 0:
            print("No test users evaluated.")
            return ({}, per_user) if return_per_user else {}
        self._print_mimic_ranking_aggregate(
            n_ev, sum_auc, sum_prec, sum_rec, sum_ndcg, sum_hit, sum_mrr,
            title="MIMIC-IV test (final)",
        )
        out = {
            "n_eval":                n_ev,
            "auc":                   float(sum_auc / n_ev),
            "precision":             [float(v / n_ev) for v in sum_prec],
            "recall":                [float(v / n_ev) for v in sum_rec],
            "ndcg":                  [float(v / n_ev) for v in sum_ndcg],
            "hit":                   [float(v / n_ev) for v in sum_hit],
            "mrr":                   [float(v / n_ev) for v in sum_mrr],
            "ks":                    list(ks),
            "inference_time_s":      round(inference_time_s, 4),
            "inference_throughput":  round(self.n_users / inference_time_s, 1) if inference_time_s > 0 else 0,
        }
        return (out, per_user) if return_per_user else out

    def fit(self):
        """
        Fitting a neural network with early stopping.
        """
        if getattr(self.args, "dataset", "cora") in ("mimic", "eicu"):
            return self.fit_mimic()
        accuracy = 0
        no_improvement = 0
        epochs = trange(self.args.epochs, desc="Accuracy")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        for _ in epochs:
            self.optimizer.zero_grad()
            prediction = self.model(self.propagation_matrix, self.features)
            loss = torch.nn.functional.nll_loss(prediction[self.train_nodes], self.target[self.train_nodes])
            if self.args.model == "mixhop" and self.base_run == True:
                loss = loss + self.model.calculate_group_loss()
            elif self.args.model == "mixhop" and self.base_run == False:
                loss = loss + self.model.calculate_loss()
            loss.backward()
            self.optimizer.step()
            new_accuracy = self.score(self.validation_nodes)
            epochs.set_description("Validation Accuracy: %g" % round(new_accuracy, 4))
            if new_accuracy < accuracy:
                no_improvement = no_improvement + 1
                if no_improvement == self.args.early_stopping:
                    epochs.close()
                    break
            else:
                no_improvement = 0
                accuracy = new_accuracy
        metrics = self.classification_metrics(self.test_nodes)
        print("\nTest metrics:")
        print("  Precision (macro): %.4f" % metrics["precision_macro"])
        print("  Recall (macro): %.4f" % metrics["recall_macro"])
        print("  AUC (ovr-macro): %.4f" % metrics["auc_ovr_macro"])
        print("  NDCG@5: %.4f" % metrics["ndcg_at_5"])
        print("  Hit Rate@5: %.4f\n" % metrics["hit_rate_at_5"])

    def score(self, indices):
        """
        Scoring a neural network.
        :param indices: Indices of nodes involved in accuracy calculation.
        :return acc: Accuracy score.
        """
        self.model.eval()
        _, prediction = self.model(self.propagation_matrix, self.features).max(dim=1)
        correct = prediction[indices].eq(self.target[indices]).sum().item()
        acc = correct / indices.shape[0]
        return acc

    def classification_metrics(self, indices):
        """
        Compute classification metrics on selected node indices.
        :param indices: node indices for evaluation.
        :return: dict with precision/recall/auc/ndcg/hit-rate.
        """
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(self.propagation_matrix, self.features)
            probs = torch.exp(logits)
            _, prediction = logits.max(dim=1)
        y_true = self.target[indices].detach().cpu().numpy()
        y_pred = prediction[indices].detach().cpu().numpy()
        y_prob = probs[indices].detach().cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        try:
            present = np.unique(y_true)
            if present.size < 2:
                auc = 0.0
            else:
                y_true_bin = np.zeros((y_true.shape[0], y_prob.shape[1]), dtype=np.int32)
                y_true_bin[np.arange(y_true.shape[0]), y_true] = 1
                auc = float(
                    roc_auc_score(
                        y_true_bin[:, present],
                        y_prob[:, present],
                        average="macro",
                    )
                )
        except Exception:
            auc = 0.0
        topk = 20
        topk_idx = np.argpartition(-y_prob, topk - 1, axis=1)[:, :topk]
        hit = (topk_idx == y_true[:, None]).any(axis=1).astype(np.float32)
        hit_rate_at_5 = float(np.mean(hit))
        # Single-label NDCG@K: 1/log2(rank+1) if true class in top-k else 0.
        ndcgs = []
        for i in range(y_prob.shape[0]):
            ranked = np.argsort(-y_prob[i])[:topk]
            pos = np.where(ranked == y_true[i])[0]
            if pos.size == 0:
                ndcgs.append(0.0)
            else:
                ndcgs.append(1.0 / np.log2(float(pos[0]) + 2.0))
        ndcg_at_5 = float(np.mean(ndcgs))
        return {
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "auc_ovr_macro": auc,
            "ndcg_at_5": ndcg_at_5,
            "hit_rate_at_5": hit_rate_at_5,
        }

    def evaluate_architecture(self):
        """
        Making a choice about the optimal layer sizes.
        """
        print("The best architecture is:\n")
        self.layer_sizes = dict()

        self.layer_sizes["upper"] = []

        for layer in self.model.upper_layers:
            norms = torch.norm(layer.weight_matrix**2, dim=0)
            norms = norms[norms < self.args.cut_off]
            self.layer_sizes["upper"].append(norms.shape[0])

        self.layer_sizes["bottom"] = []

        for layer in self.model.bottom_layers:
            norms = torch.norm(layer.weight_matrix**2, dim=0)
            norms = norms[norms < self.args.cut_off]
            self.layer_sizes["bottom"].append(norms.shape[0])

        self.layer_sizes["upper"] = [int(self.args.budget*layer_size/sum(self.layer_sizes["upper"]))  for layer_size in self.layer_sizes["upper"]]
        self.layer_sizes["bottom"] = [int(self.args.budget*layer_size/sum(self.layer_sizes["bottom"]))  for layer_size in self.layer_sizes["bottom"]]
        print("Layer 1.: "+str(tuple(self.layer_sizes["upper"])))
        print("Layer 2.: "+str(tuple(self.layer_sizes["bottom"])))

    def reset_architecture(self):
        """
        Changing the layer sizes.
        """
        print("\nResetting the architecture.\n")
        self.args.layers_1 = self.layer_sizes["upper"]
        self.args.layers_2 = self.layer_sizes["bottom"]
        return self.args
