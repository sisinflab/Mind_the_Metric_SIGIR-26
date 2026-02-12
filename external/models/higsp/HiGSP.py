import torch
import numpy as np
from scipy.sparse import csr_matrix
import scipy
import gc
from tqdm import tqdm
import scipy.sparse as sp
import time
from sklearn.mixture import GaussianMixture
import sys
from scipy.sparse.linalg import svds
import gc

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class HiGSP(RecMixin, BaseRecommenderModel):
    r"""
    Hierarchical Graph Signal Processing for Collaborative Filtering

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3589334.3645368>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
            ("_pri_factor", "pri_factor", "pri_factor", 80, int, None),
            ("_alpha_1", "alpha_1", "alpha_1", 0.08, float, None),
            ("_alpha_2", "alpha_2", "alpha_2", 0.73, float, None),
            ("_order1", "order1", "order1", 2, int, None),
            ("_order2", "order2", "order2", 12, int, None),
            ("_n_clusters", "n_clusters", "n_clusters", 25, int, None),
        ]

        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        row, col = data.sp_i_train.nonzero()
        self.adj_mat = np.zeros(data.sp_i_train.shape, dtype=np.float64)
        self.adj_mat[row, col] = 1.0

    def matpow(self, mat, order):
        R = mat
        if order == 1:
            return R
        for _ in range(2, order + 1):
            R = R.T @ mat
            gc.collect()
        return R

    def normalize_adj_mat(self, adj_mat):
        adj_mat = sp.csr_matrix(adj_mat)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.A
        return norm_adj

    def normalize_adj_mat_sp(self, adj_mat):
        adj_mat = sp.csr_matrix(adj_mat)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_i_inv = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        return norm_adj, d_mat_i, d_mat_i_inv

    def construct_cluster_wise_filter(self, adj_mat):
        # Cluster users based on their interactions
        clustering = GaussianMixture(n_components=self._n_clusters, verbose=2, covariance_type='diag')
        print("Fitting clustering...")
        start = time.time()
        cluster_labels = clustering.fit_predict(adj_mat)
        del clustering
        gc.collect()
        end = time.time()
        self.logger.info(f"Time for clustering: {end - start}")

        gc.collect()

        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        self.logger.info(f"Number of clusters: {n_clusters}")

        I = np.eye(self._num_items, dtype=np.float32)
        local_filter = np.empty(
            (n_clusters, self._num_items, self._num_items),
            dtype=np.float32
        )

        # Itera sui cluster uno alla volta per risparmiare memoria
        # Invece di costruire il tensore gigante [K, U, I] e [K, I, I]
        for idx, cluster_id in enumerate(tqdm(unique_clusters)):
            # 1. Seleziona solo gli utenti appartenenti al cluster corrente
            mask = (cluster_labels == cluster_id)
            C_tilde = adj_mat[mask, :]
            # 2. Calcola A_tilde per il singolo cluster
            # A_tilde = C^T * C
            A_tilde = C_tilde.T @ C_tilde
            A_tilde = self.normalize_adj_mat(A_tilde)
            # 3. Calcola L_tilde = I - A_tilde
            L_tilde = I - A_tilde
            # 4. Calcola la potenza della matrice (L_tilde)^k
            # Usiamo matpow (versione singola matrice) invece di bmatpow (versione batch)
            # Questo sposta su GPU/CPU solo una matrice [I, I] alla volta invece di [K, I, I]
            L_tilde_k = self.matpow(L_tilde, self._order1)
            # 5. Calcola il filtro finale per questo cluster
            # Filter = I - L^k
            local_filter[idx] = I - L_tilde_k
            gc.collect()

        return local_filter, cluster_labels

    def construct_global_aware_filter(self, adj_mat):
        # Construct ideal low-pass filter
        norm_adj, d_mat_i, d_mat_i_inv = self.normalize_adj_mat_sp(adj_mat)
        norm_adj = norm_adj.tocsc()
        # ut, s, vt = np.linalg.svd(norm_adj, self._pri_factor)
        k = min(self._pri_factor, min(norm_adj.shape) - 1)
        _, _, vt = svds(norm_adj, k=k)
        # vt = np.flip(vt, axis=0)
        global_filter1 = d_mat_i @ vt.T @ vt @ d_mat_i_inv

        # Construct high-order low-pass filter
        R_tilde = self.normalize_adj_mat(adj_mat)
        P_tilde = R_tilde.T @ R_tilde
        L_tilde = np.identity(P_tilde.shape[1]) - P_tilde
        L_tilde_k = self.matpow(L_tilde, self._order2)
        global_filter2 = np.identity(P_tilde.shape[1]) - L_tilde_k
        return global_filter1, global_filter2

    def train(self):
        start = time.time()

        self.logger.info(f"Construct item-wise filter")
        self.item_cluster_filter, self.item_cluster_labels = self.construct_cluster_wise_filter(self.adj_mat)

        self.logger.info(f"Construct globally-aware filter")
        self.global_filter1, self.global_filter2 = self.construct_global_aware_filter(self.adj_mat)

        end = time.time()
        self.logger.info(f"Training has taken: {end - start}")

        gc.collect()

        self.evaluate()

    def get_users_rating(self, batch_start, batch_stop):
        # 1. Retrieve the adjacency matrix slice for the current batch of users
        batch_adj = self.adj_mat[batch_start:batch_stop]
        current_batch_size = batch_adj.shape[0]
        n_items = self.adj_mat.shape[1]
        n_clusters = self.item_cluster_filter.shape[0]

        # 2. Compute Cluster-wise predictions for this batch
        # Retrieve cluster labels for the current batch of users
        batch_labels = self.item_cluster_labels[batch_start:batch_stop]

        # Create the C tensor ONLY for the current batch to save memory
        # Shape: (n_clusters, batch_size, n_items)
        C_batch = np.zeros((n_clusters, current_batch_size, n_items), dtype=np.float32)

        # Place the user interactions into the correct cluster slice
        # Vectorized assignment: C_batch[cluster_id, user_index_in_batch, :] = user_interactions
        C_batch[batch_labels, np.arange(current_batch_size), :] = batch_adj

        # item_cluster_filter shape: (K, I, I)
        # Result: (K, B, I)
        batch_ratings_cluster = np.matmul(C_batch, self.item_cluster_filter)

        # Sum over clusters → (B, I)
        batch_ratings = np.sum(batch_ratings_cluster, axis=0)

        # 3. Compute Global-aware predictions for this batch
        # Add contributions from global filters
        # (Batch, Items) x (Items, Items) -> (Batch, Items)
        batch_ratings += self._alpha_1 * (batch_adj @ self.global_filter1)
        batch_ratings += self._alpha_2 * (batch_adj @ self.global_filter2)

        return batch_ratings


    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(tqdm(range(0, self._num_users, self._batch_eval))):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = self.get_users_rating(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), torch.tensor(preds).to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    @property
    def name(self):
        return "HiGSP" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"