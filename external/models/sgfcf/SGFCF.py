import torch
import numpy as np
import random
from tqdm import tqdm
import gc
from time import time

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class SGFCF(RecMixin, BaseRecommenderModel):
    r"""
    How Powerful is Graph Filtering for Recommendation?

    For further details, please refer to the `paper <http://arxiv.org/abs/2406.08827>`_
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_alpha", "alpha", "alpha", 0.3, float, None),

            ("_beta_1", "beta_1", "b_1", 1.0, float, None),
            ("_beta_2", "beta_2", "b_2", 1.0, float, None),

            ("_gamma", "gamma", "gamma", 1.0, float, None),
            ("_eps", "eps", "eps", 1.0, float, None)
        ]

        self.autoset_params()

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        row, col = data.sp_i_train.nonzero()
        self.freq_matrix = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.array([row, col])),
            values=torch.FloatTensor(np.ones_like(row)),
            size=(self._num_users, self._num_items), dtype=torch.float
        ).coalesce().to(self.device)

    def train(self):
        start = time()

        homo_ratio_user, homo_ratio_item = self._homo_ratio()
        gc.collect()
        torch.cuda.empty_cache()

        D_u = 1 / (torch.sparse.sum(self.freq_matrix, dim=1).to_dense() + self._alpha).pow(self._eps)
        D_i = 1 / (torch.sparse.sum(self.freq_matrix, dim=0).to_dense() + self._alpha).pow(self._eps)
        D_u[D_u == float('inf')] = 0
        D_i[D_i == float('inf')] = 0
        vals = self.freq_matrix.values()
        rows, cols = self.freq_matrix.indices()
        vals.mul_(D_u[rows]).mul_(D_i[cols])
        norm_freq_matrix = torch.sparse_coo_tensor(
            indices=torch.stack([rows, cols]),
            values=vals,
            size=self.freq_matrix.size(),
            device=self.device
        ).coalesce()

        del D_u, D_i
        gc.collect()
        torch.cuda.empty_cache()

        U, value, V = torch.svd_lowrank(norm_freq_matrix, q=self._factors + 200, niter=30)
        value.div_(value.max())

        def individual_weight(value, homo_ratio):
            y_min, y_max = self._beta_1, self._beta_2
            x_min, x_max = homo_ratio.min(), homo_ratio.max()
            homo_weight = (y_max - y_min) / (x_max - x_min) * homo_ratio + (x_max * y_min - y_max * x_min) / (
                        x_max - x_min)
            homo_weight = homo_weight.unsqueeze(1)
            return value.pow(homo_weight)

        V_weighted_t = (V[:, :self._factors] * individual_weight(value[:self._factors], homo_ratio_item)).t()

        del V, homo_ratio_item
        gc.collect()
        torch.cuda.empty_cache()

        self.rate_matrix = torch.empty(self._num_users, self._num_items, device=self.device)

        # batching the instruction:
        # self.rate_matrix = (U[:, :self._factors] * individual_weight(value[:self._factors], homo_ratio_user)).mm(
        #             (V[:, :self._factors] * individual_weight(value[:self._factors], homo_ratio_item)).t())

        for i in tqdm(range(0, self._num_users, self._batch_eval), desc="Reconstructing rate matrix",
                      disable=not self._verbose):
            stop_idx = min(i + self._batch_eval, self._num_users)
            batch_users = torch.arange(i, stop_idx).to(self.device)
            user_batch_U = U[i:stop_idx, :self._factors]
            user_batch_homo = homo_ratio_user.index_select(dim=0, index=batch_users) # homo_ratio_user[i:stop_idx]
            user_batch_weighted = user_batch_U * individual_weight(value[:self._factors], user_batch_homo)
            self.rate_matrix[i:stop_idx] = user_batch_weighted.mm(V_weighted_t)

        del U, value, homo_ratio_user, V_weighted_t
        gc.collect()
        torch.cuda.empty_cache()

        # self.rate_matrix = self.rate_matrix / (self.rate_matrix.sum(1).unsqueeze(1))
        row_sums = self.rate_matrix.sum(1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        self.rate_matrix.div_(row_sums)

        del row_sums
        gc.collect()
        torch.cuda.empty_cache()

        # norm_freq_matrix = norm_freq_matrix.mm(norm_freq_matrix.t()).mm(norm_freq_matrix)
        high_order_matrix = torch.empty(self._num_users, self._num_items, device=self.device)
        for i in tqdm(range(0, self._num_users, self._batch_eval), desc="Computing high-order proximity", disable=not self._verbose):
            stop_idx = min(i + self._batch_eval, self._num_users)
            batch_users = torch.arange(i, stop_idx).to(self.device)
            user_batch = norm_freq_matrix.index_select(dim=0, index=batch_users) # norm_freq_matrix[i:stop_idx]  # [batch_size, num_items]
            # (A_batch @ A.T) @ A
            user_user_batch_similarity = torch.sparse.mm(user_batch, norm_freq_matrix.t()) # user_batch.mm(norm_freq_matrix.t())  # Dim: [batch_size, num_users]
            batch_result = torch.sparse.mm(user_user_batch_similarity, norm_freq_matrix).to_dense() # user_user_batch_similarity.mm(norm_freq_matrix)  # Dim: [batch_size, num_items]
            high_order_matrix[i:stop_idx] = batch_result
        norm_freq_matrix = high_order_matrix
        del high_order_matrix
        gc.collect()
        torch.cuda.empty_cache()

        # norm_freq_matrix = norm_freq_matrix / (norm_freq_matrix.sum(1).unsqueeze(1))
        row_sums = norm_freq_matrix.sum(1, keepdim=True)
        row_sums[row_sums == 0] = 1.0  # Protezione per la divisione per zero
        norm_freq_matrix.div_(row_sums)  # divisione IN-PLACE per risparmiare memoria

        # self.rate_matrix = (self.rate_matrix + self._gamma * norm_freq_matrix).sigmoid()
        torch.add(self.rate_matrix, norm_freq_matrix, alpha=self._gamma, out=self.rate_matrix)
        torch.sigmoid(self.rate_matrix, out=self.rate_matrix)

        # self.rate_matrix = self.rate_matrix - self.freq_matrix * 1000  # masking in evaluation

        del self.freq_matrix, norm_freq_matrix, row_sums
        gc.collect()
        torch.cuda.empty_cache()

        end = time()
        self.logger.info(f"Training has taken: {end - start}")

        self.evaluate()

    def _homo_ratio(self):
        train_data = [[] for i in range(self._num_users)]
        train_data_item = [[] for i in range(self._num_items)]

        user_idx, item_idx = self.freq_matrix.indices()
        for u, i in zip(user_idx.tolist(), item_idx.tolist()):
            train_data[u].append(i)
            train_data_item[i].append(u)

        homo_ratio_user, homo_ratio_item = [], []
        for u in tqdm(range(self._num_users), desc="Computing homo_ratio_user", disable=not self._verbose):
            if len(train_data[u]) > 1:
                items_u = torch.tensor(train_data[u], device=self.device)
                inter_items = self.freq_matrix.index_select(dim=1, index=items_u).t().coalesce()
                rows, cols = inter_items.indices()
                vals = inter_items.values()
                mask = cols != u
                inter_items = torch.sparse_coo_tensor(
                    indices=torch.stack([rows[mask], cols[mask]]),
                    values=vals[mask],
                    size=inter_items.size(),
                    device=inter_items.device
                ).coalesce()
                connect_matrix = torch.sparse.mm(inter_items, inter_items.t())
                rows, cols = connect_matrix.indices()
                mask = rows != cols
                nnz_offdiag = mask.sum().item()
                size = inter_items.shape[0]
                ratio_u = nnz_offdiag / (size * (size - 1))
                homo_ratio_user.append(ratio_u)
            else:
                homo_ratio_user.append(0)
        for i in tqdm(range(self._num_items), desc="Computing homo_ratio_item", disable=not self._verbose):
            if len(train_data_item[i]) > 1:
                users_i = torch.tensor(train_data_item[i], device=self.device)
                inter_users = self.freq_matrix.index_select(dim=0, index=users_i).coalesce()
                rows, cols = inter_users.indices()
                vals = inter_users.values()
                mask = cols != i
                inter_users = torch.sparse_coo_tensor(
                    indices=torch.stack([rows[mask], cols[mask]]),
                    values=vals[mask],
                    size=inter_users.size(),
                    device=inter_users.device
                ).coalesce()
                connect_matrix = torch.sparse.mm(inter_users, inter_users.t())
                rows, cols = connect_matrix.indices()
                mask = rows != cols
                nnz_offdiag = mask.sum().item()
                size = inter_users.shape[0]
                ratio_i = nnz_offdiag / (size * (size - 1))
                homo_ratio_item.append(ratio_i)
            else:
                homo_ratio_item.append(0)

        homo_ratio_user = torch.Tensor(homo_ratio_user).to(self.device)
        homo_ratio_item = torch.Tensor(homo_ratio_item).to(self.device)
        return homo_ratio_user, homo_ratio_item

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(tqdm(range(0, self._num_users, self._batch_eval), disable=not self._verbose, desc="Evaluating")):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = self.get_users_rating(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_users_rating(self, batch_start, batch_stop):
        return self.rate_matrix[batch_start:batch_stop]

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
        return "SGFCF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"