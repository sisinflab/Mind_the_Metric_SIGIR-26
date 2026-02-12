import torch
import numpy as np
from scipy.sparse import csr_matrix
import scipy
from tqdm import tqdm
import scipy.sparse as sp
import time
import gc

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class FaGSP(RecMixin, BaseRecommenderModel):
    r"""
    Frequency-aware Graph Signal Processing for Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/2402.08426>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
            ("_pri_factor1", "pri_factor1", "pri_factor1", 256, int, None),
            ("_pri_factor2", "pri_factor2", "pri_factor2", 128, int, None),
            ("_alpha_1", "alpha_1", "alpha_1", 0.3, float, None),
            ("_alpha_2", "alpha_2", "alpha_2", 0.5, float, None),
            ("_order1", "order1", "order1", 12, int, None),
            ("_order2", "order2", "order2", 14, int, None),
            ("_q", "q", "q", 0.7, float, None)
        ]

        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        row, col = data.sp_i_train.nonzero()
        self.rat_mat = csr_matrix((np.ones(len(row)), (row, col)), shape=(self._num_users, self._num_items))

        self.adj_mat = data.sp_i_train.tolil()
        self.d_mat_i = None
        self.d_mat_i_inv = None
        self.vt1 = None
        self.vt2 = None
        self.RTR1_pow = None
        self.RTR2_pow = None
        self.quan = None

    def matpow(self, mat, order):
        R = mat
        if order == 1:
            return R
        for ord in range(2, order+1):
            R = R.T @ mat
        return R

    def train(self):
        start = time.time()

        adj_mat = self.adj_mat
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv_row = np.power(rowsum, -0.5).flatten()
        d_inv_row[np.isinf(d_inv_row)] = 0.
        d_mat_row = scipy.sparse.diags(d_inv_row)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv_col = np.power(colsum, -0.5).flatten()
        d_inv_col[np.isinf(d_inv_col)] = 0.
        self.d_mat_i = scipy.sparse.diags(d_inv_col)
        self.d_mat_i_inv = scipy.sparse.diags(1 / d_inv_col)

        norm_adj = d_mat_row.dot(adj_mat)
        norm_adj = norm_adj.dot(self.d_mat_i)
        norm_adj_csc = norm_adj.tocsc()

        _, _, vt_full = np.linalg.svd(norm_adj_csc.A, full_matrices=False)

        # VT2 (Low-pass High-freq components? Logic from original: takes last k)
        self.vt2 = vt_full[-self._pri_factor2:]
        # VT1 (Top-k components)
        self.vt1 = vt_full[:self._pri_factor1]

        # --- Filter Powers ---
        # Item-Item filter
        RTR1 = norm_adj_csc.T @ norm_adj_csc
        self.RTR1_pow = self.matpow(np.eye(RTR1.shape[0]) - RTR1, self._order1)
        # User-User filter
        RTR2 = norm_adj_csc @ norm_adj_csc.T
        self.RTR2_pow = self.matpow(np.eye(RTR2.shape[0]) - RTR2, self._order2)

        self.logger.info("Computing Quantiles (Temporary Dense Step)...")
        # We need to compute P30 once to get the quantiles, then we can discard it.
        # Constructing full dense rating matrix just for this step
        batch_test_full = self.rat_mat.toarray()
        P30_temp = batch_test_full @ self.d_mat_i @ self.vt2.T @ self.vt2 @ self.d_mat_i_inv
        self.quan = np.quantile(P30_temp, q=self._q, axis=0, keepdims=True)

        del P30_temp
        del batch_test_full
        del norm_adj_csc
        del vt_full
        gc.collect()

        end = time.time()
        self.logger.info(f"Training has taken: {end - start}")
        self.evaluate()

    def get_users_rating(self, batch_start, batch_stop):
        batch_rat = self.rat_mat[batch_start:batch_stop].toarray()          # (Batch, Items)

        batch_preds = np.zeros_like(batch_rat)          # Initialization

        # --- P11: Item-side High-pass ---
        # P11 = R @ (I - RTR1_pow)
        P11 = batch_rat @ (np.eye(self.RTR1_pow.shape[0]) - self.RTR1_pow)      # (Batch, I) @ (I, I) -> (Batch, I)
        batch_preds += P11

        # --- P12: User-side High-pass ---
        # P12 = (I - RTR2_pow)_slice @ R_full
        # We need the specific rows of the user-user filter corresponding to this batch
        # (Batch, U) @ (U, I) -> (Batch, I)
        filter_slice = np.eye(self.RTR2_pow.shape[0])[batch_start:batch_stop] - self.RTR2_pow[batch_start:batch_stop]
        P12 = filter_slice @ self.rat_mat
        batch_preds += P12

        # --- P30 Reconstruction for Batch ---
        # P30 = R_batch @ d_mat_i @ vt2.T @ vt2 @ d_mat_i_inv
        P30_batch = batch_rat @ self.d_mat_i @ self.vt2.T @ self.vt2 @ self.d_mat_i_inv

        # Apply Quantile Filtering (using cached quantiles)
        # Create a mask for thresholding
        mask_high = P30_batch > self.quan
        P30_batch[mask_high] = 1.0
        P30_batch[~mask_high] = 0.0

        P30_batch[batch_rat < 1] = 0.0          # Apply Interaction Mask (Keep only where user rated)

        # --- P3 Construction ---
        P3_batch = batch_rat + self._alpha_2 * P30_batch

        # --- P2: Low-pass on P3 ---
        # P2 = P3 @ d_mat_i @ vt1.T @ vt1 @ d_mat_i_inv
        # Note: The scaling matrices (d_mat_i, etc.) are the same as used for adj_mat
        # based on the analysis of the original code's variable reuse.
        P2 = P3_batch @ self.d_mat_i @ self.vt1.T @ self.vt1 @ self.d_mat_i_inv

        batch_preds += self._alpha_1 * P2

        return batch_preds

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
        return "FaGSP" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"