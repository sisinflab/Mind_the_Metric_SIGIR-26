import numpy as np
from operator import itemgetter
from scipy import sparse as sp
from sklearn.utils.extmath import randomized_svd

from elliot.utils import logging as logging_project
logger = logging_project.get_logger("__main__")

class PureSVDModel(object):

    def __init__(self, factors, data, random_seed):
        np.random.seed(random_seed)

        self._data = data
        self._private_users = data.private_users
        self._public_users = data.public_users
        self._private_items = data.private_items
        self._public_items = data.public_items
        self.factors = factors
        self.random_seed = random_seed

        self.user_vec, self.item_vec = None, None
        self._preds = None

    def initialize(self):
        # Computing SVD
        U, sigma, Vt = randomized_svd(self._data.sp_i_train, n_components=self.factors, random_state=self.random_seed)

        # similarity matrix
        self.W_sparse = np.matmul(np.transpose(Vt), Vt)

        # predictions
        self._preds = self._data.sp_i_train_ratings.dot(self.W_sparse)

    def get_user_recs_batch(self, u, mask, k):
        u_index = itemgetter(*u)(self._data.public_users)
        users_recs = np.where(mask[u_index, :], self._preds[u_index, :], -np.inf)
        index_ordered = np.argpartition(users_recs, -k, axis=1)[:, -k:]
        value_ordered = np.take_along_axis(users_recs, index_ordered, axis=1)
        local_top_k = np.take_along_axis(index_ordered, value_ordered.argsort(axis=1)[:, ::-1], axis=1)
        value_sorted = np.take_along_axis(users_recs, local_top_k, axis=1)
        mapper = np.vectorize(self._data.private_items.get)
        return [[*zip(item, val)] for item, val in zip(mapper(local_top_k), value_sorted)]