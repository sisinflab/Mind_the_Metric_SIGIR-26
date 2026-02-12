"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np
from scipy import sparse as sp
from sklearn.utils.extmath import randomized_svd
from operator import itemgetter


class RSVDModel(object):

    def __init__(self, factors, reg, data, random_seed):

        self._data = data
        self._private_users = data.private_users
        self._public_users = data.public_users
        self._private_items = data.private_items
        self._public_items = data.public_items
        self.factors = factors
        self.reg = reg
        self.random_seed = random_seed
        self.train_dict = self._data.train_dict
        self.user_num, self.item_num = self._data.num_users, self._data.num_items

        self.user_vec, self.item_vec = None, None

    def train(self):
        U, sigma, Vt = randomized_svd(self._data.sp_i_train,
                                      n_components=self.factors,
                                      random_state=self.random_seed)
        self.user_vec = U
        sigma_squared_minus_reg = np.maximum(0, sigma ** 2 - self.reg)
        omega_diag = np.sqrt(sigma_squared_minus_reg)
        self.item_vec = Vt.T * omega_diag

    def predict(self, user):
        return self.user_vec[user, :].dot(self.item_vec.T)

    def get_user_recs(self, u, mask, k=100):
        u_index = itemgetter(*u)(self._data.public_users)
        preds = self.predict(u_index)
        users_recs = np.where(mask[u_index, :], preds, -np.inf)
        index_ordered = np.argpartition(users_recs, -k, axis=1)[:, -k:]
        value_ordered = np.take_along_axis(users_recs, index_ordered, axis=1)
        local_top_k = np.take_along_axis(index_ordered, value_ordered.argsort(axis=1)[:, ::-1], axis=1)
        value_sorted = np.take_along_axis(users_recs, local_top_k, axis=1)
        mapper = np.vectorize(self._data.private_items.get)
        return [[*zip(item, val)] for item, val in zip(mapper(local_top_k), value_sorted)]