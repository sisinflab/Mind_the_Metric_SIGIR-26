import time
import random
import sys

import torch
import numpy as np
from tqdm import tqdm

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

from .sparse_matmul import batch_dense_matmul

class TurboCF(RecMixin, BaseRecommenderModel):
    r"""
    Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation

    For further details, please refer to the `paper <https://arxiv.org/abs/2404.14243>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml


    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 512, int, None),
            ("_alpha", "alpha", "alpha", 0.3, float, None),
            ("_power", "power", "power", 1.0, float, None),
            ("_filter", "filter", "filter", 1, int, None),
            ("_seed", "seed", "seed", 42, int, None),
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

        if self._filter < 1 or self._filter > 3:
            raise ValueError(f"chosen between: (1) linear, (2) 2nd order, (3) polynomial !!!!!!")

        # compute user-item interaction matrix
        row, col = data.sp_i_train.nonzero()
        self.R = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.array([row, col])),
            values=torch.FloatTensor(np.ones_like(row, dtype=np.float64)),
            size=(self._num_users, self._num_items),
            dtype=torch.float
        ).coalesce().to(self.device)

        self.LPF = None


    def normalized_sparse_rating_matrix(self, m, alpha):
        rowsum = torch.sparse.sum(m, dim=1).to_dense()
        rowsum = torch.pow(rowsum, -alpha)

        colsum = torch.sparse.sum(m, dim=0).to_dense()
        colsum = torch.pow(colsum, alpha - 1)

        indices = m.coalesce().indices()
        values = m.coalesce().values()

        row_factors = rowsum[indices[0]]
        col_factors = colsum[indices[1]]
        values = values * row_factors * col_factors

        R_tilde = torch.sparse_coo_tensor(indices, values, m.shape, device=self.device)

        return R_tilde.coalesce()

    def train(self):
        start = time.time()

        R_tilde = self.normalized_sparse_rating_matrix(self.R, self._alpha).to(self.device)
        P = (R_tilde.T @ R_tilde).to_dense()
        P.pow_(self._power)

        del R_tilde
        torch.cuda.empty_cache()

        if self._filter == 1:
            self.LPF = (P)

        elif self._filter == 2:
            # self.LPF = (2 * P - PP)
            PP = batch_dense_matmul(A=P, B=P, device=self.device, batch_size=1000)
            self.LPF = 2 * P
            self.LPF.sub_(PP)

            del PP
            torch.cuda.empty_cache()

        elif self._filter == 3:
            # self.LPF = (P + 0.01 * (-P @ P @ P + 10 * P @ P - 29 * P))
            # P @ P
            PP = batch_dense_matmul(A=P, B=P, device=self.device, batch_size=1000)
            # P @ P @ P
            PPP = batch_dense_matmul(A=PP, B=P, device=self.device, batch_size=1000)
            self.LPF = (P + 0.01 * (-PPP + 10 * PP - 29 * P))

        end = time.time()
        logger.info(f"The similarity computation has taken: {end - start}")

        self.evaluate()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(tqdm(range(0, self._num_users, self._batch_eval))):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = self.predict(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def predict(self, start, stop):
        batch = torch.arange(start, stop).to(self.device)
        user = self.R.index_select(dim=0, index=batch).to_dense()
        return torch.sparse.mm(self.LPF, user.T).T

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
        return "TurboCF" \
               + f"_{self.get_params_shortcut()}"
