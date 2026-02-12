"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time
from operator import itemgetter

import numpy as np
import scipy
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm
import torch

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from similaripy import similarity


class EASER(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_neighborhood", "neighborhood", "neighborhood", -1, int, None),
            ("_l2_norm", "l2_norm", "l2_norm", 1e3, float, None)
        ]

        self.autoset_params()
        if self._neighborhood == -1:
            self._neighborhood = self._data.num_items

    @property
    def name(self):
        return f"EASER_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k):
        # return {u: self.get_user_predictions(u, mask, k) for u in self._data.train_dict.keys()}
        recs = {}
        for i in tqdm(range(0, len(self._data.train_dict.keys()), 1024), desc="Processing batches",
                      total=len(self._data.train_dict.keys()) // 1024 + (1 if len(self._data.train_dict.keys()) % 1024 != 0 else 0)):
            batch = list(self._data.train_dict.keys())[i:i + 1024]
            mat = self.get_user_recs_batch(batch, mask, k)
            proc_batch = dict(zip(batch, mat))
            recs.update(proc_batch)
        return recs

    def get_user_recs_batch(self, u, mask, k):
        u_index = itemgetter(*u)(self._data.public_users)
        if torch.cuda.is_available():
            train_batch = self._train[u_index, :]
            train_batch = torch.sparse_coo_tensor(
                train_batch.nonzero(),
                train_batch.data,
                train_batch.shape,
                dtype=torch.float32,
                device="cuda"
            )
            preds = torch.sparse.mm(train_batch, self._similarity_matrix)
            preds = preds.cpu().numpy()
        else:
            preds = safe_sparse_dot(self._train[u_index, :], self._similarity_matrix).toarray()

        users_recs = np.where(mask[u_index, :], preds, -np.inf)
        index_ordered = np.argpartition(users_recs, -k, axis=1)[:, -k:]
        value_ordered = np.take_along_axis(users_recs, index_ordered, axis=1)
        local_top_k = np.take_along_axis(index_ordered, value_ordered.argsort(axis=1)[:, ::-1], axis=1)
        value_sorted = np.take_along_axis(users_recs, local_top_k, axis=1)
        mapper = np.vectorize(self._data.private_items.get)
        return [[*zip(item, val)] for item, val in zip(mapper(local_top_k), value_sorted)]

    def to_scipy_csr(self, sparse_tensor):
        """Converts a PyTorch sparse tensor to a SciPy CSR matrix."""
        if not sparse_tensor.is_coalesced():
            sparse_tensor = sparse_tensor.coalesce()  # Ensure COO format is coalesced

        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        shape = sparse_tensor.shape

        return scipy.sparse.csr_matrix((values.numpy(), (indices[0].numpy(), indices[1].numpy())), shape=shape)

    def train(self):
        if self._restore:
            return self.restore_weights()


        start = time.time()

        self._train = self._data.sp_i_train_ratings

        # self._similarity_matrix = safe_sparse_dot(self._train.T, self._train, dense_output=False)
        self._similarity_matrix = similarity.dot_product(self._train.T, self._train,
                                                         k=self._train.shape[0], format_output= 'csr')


        diagonal_indices = np.diag_indices(self._similarity_matrix.shape[0])
        item_popularity = np.ediff1d(self._train.tocsc().indptr)
        self._similarity_matrix.setdiag(item_popularity + self._l2_norm)

        if torch.cuda.is_available():
            self.logger.info(f"Use CUDA for Inverse")
            self._similarity_matrix = torch.tensor(data=self._similarity_matrix.todense(),
                                                   dtype=torch.float32).cuda()
            torch.cuda.synchronize()
            S = np.zeros((self._data.num_items, self._data.num_items))
            I = torch.eye(self._data.num_items).cuda()
            batch_size=1024
            for start in tqdm(range(0, self._data.num_items, batch_size), disable=False):
                end = min(start + batch_size, self._data.num_items)
                block = I[:, start:end]
                X = torch.linalg.solve(self._similarity_matrix, block)
                X = X.cpu().numpy()
                diag_vals = -np.diag(X[start:end, :])
                S[:, start:end] = X / diag_vals
            np.fill_diagonal(S, 0)
            self._similarity_matrix = S
            torch.cuda.empty_cache()
        else:
            self.logger.info(f"Classical Inverse")
            P = np.linalg.inv(self._similarity_matrix.todense())
            self._similarity_matrix = P / (-np.diag(P))
            self._similarity_matrix[diagonal_indices] = 0.0

        end = time.time()
        self.logger.info(f"The similarity computation has taken: {end - start}")

        if torch.cuda.is_available():
            self._similarity_matrix = torch.tensor(data=self._similarity_matrix,
                                                   dtype=torch.float32).cuda()
            torch.cuda.empty_cache()
        else:
            self._similarity_matrix = scipy.sparse.csr_matrix(self._similarity_matrix)

        self.evaluate()
