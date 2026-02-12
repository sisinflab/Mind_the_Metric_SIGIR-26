import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import random

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from .sparse_matmul import batch_dense_matmul_sparse_input


class SVD_AE(RecMixin, BaseRecommenderModel):
    r"""
    SVD-AE: Simple Autoencoders for Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/2405.04746>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # set seed
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 512, int, None),
            ("_factors", "factors", "factors", 256, int, None),
        ]

        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.adj_mat = data.sp_i_train.tolil()
        self.preds = None

    def train(self):
        adj_mat = self.adj_mat
        rowsum = np.array(adj_mat.sum(axis=1))
        rowsum = np.where(rowsum == 0.0, 1.0, rowsum)
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        colsum = np.array(adj_mat.sum(axis=0))
        colsum = np.where(colsum == 0.0, 1.0, colsum)
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_i = sp.diags(d_inv)
        d_mat_i_inv = sp.diags(1 / d_inv)
        norm_adj = norm_adj.dot(d_mat_i)
        norm_adj = norm_adj.tocsc()

        del rowsum, d_inv, d_mat, colsum, d_mat_i, d_mat_i_inv

        adj_mat = self.convert_sp_mat_to_sp_tensor(adj_mat).to(self.device)
        norm_adj = self.convert_sp_mat_to_sp_tensor(norm_adj).to(self.device)

        self.logger.info("Computing svd...")
        ut, s, vt = torch.svd_lowrank(norm_adj, q=self._factors, niter=2, M=None)

        self.logger.info("Computing A...")
        A = ut @ (torch.diag(1 / s)) @ vt.T
        del ut, s, vt

        self.logger.info("Computing partial...")
        self.preds = batch_dense_matmul_sparse_input(A=A.T, B=adj_mat, device=self.device, batch_size=1000) #A.T @ adj_mat.to_dense()
        del A, adj_mat

        self.logger.info("Computing preds...")
        #self.preds = torch.mm(self.preds.T, norm_adj.T).T # torch.mm(norm_adj, self.preds)
        self.preds = batch_dense_matmul_sparse_input(A=self.preds.T, B=norm_adj.T, device=self.device, batch_size=1000)
        del norm_adj
        self.preds = self.preds.T

        self.evaluate()

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce()

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

    def predict(self, batch_start, batch_stop):
        return self.preds[batch_start:batch_stop, :]

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
        return "SVD_AE" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"