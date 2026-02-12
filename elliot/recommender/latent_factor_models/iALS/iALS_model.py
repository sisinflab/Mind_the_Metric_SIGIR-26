"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle

import numpy as np
from scipy import sparse as sp
from tqdm import tqdm


class iALSModel(object):
    """
    Simple Matrix Factorization class
    """

    def __init__(self, factors, data, random, alpha, epsilon, reg, scaling):

        self._data = data
        self.random = random
        self.C = self._data.sp_i_train
        if scaling == "linear":
            self.C.data = 1.0 + alpha * self.C.data
        elif scaling == "log":
            self.C.data = 1.0 + alpha * np.log(1.0 + self.C.data / epsilon)
        self.C_csc = self.C.tocsc()
        self.train_dict = self._data.train_dict
        self.user_num, self.item_num = self._data.num_users, self._data.num_items

        self.X = self.random.normal(scale=0.01, size=(self.user_num, factors))
        self.Y = self.random.normal(scale=0.01, size=(self.item_num, factors))

        warm_item_mask = np.ediff1d(self._data.sp_i_train.tocsc().indptr) > 0
        self.warm_items = np.arange(0, self.item_num, dtype=np.int32)[warm_item_mask]

        # self.X_eye = sp.eye(self.user_num)
        # self.Y_eye = sp.eye(self.item_num)
        self.lambda_eye = reg * sp.eye(factors)

        self.user_vec, self.item_vec, self.pred_mat = None, None, None

    def train_step(self):
        yTy = self.Y.T.dot(self.Y)

        C = self.C
        for u in tqdm(range(self.user_num), desc="Looping on users", disable=False):
            start = C.indptr[u]
            end = C.indptr[u+1]

            Cu = C.data[start:end]
            Pu = self.Y[C.indices[start:end], :]

            B = yTy + (Pu.T * (Cu - 1)) @ Pu + self.lambda_eye

            self.X[u] = np.linalg.solve(B, Pu.T.dot(Cu))

        xTx = self.X.T.dot(self.X)
        C = self.C_csc
        for i in tqdm(self.warm_items, desc="Looping on items", disable=False):
            start = C.indptr[i]
            end = C.indptr[i + 1]

            Cu = C.data[start:end]
            Pi = self.X[C.indices[start:end], :]

            B = xTx + Pi.T.dot(((Cu - 1) * Pi.T).T) + self.lambda_eye

            self.Y[i] = np.linalg.solve(B, Pi.T.dot(Cu))

    def predict(self, user, item):
        return self.pred_mat[self._data.public_users[user], self._data.public_items[item]]

    def get_user_recs(self, user, mask, k=100):
        user_id = self._data.public_users[user]
        user_mask = mask[user_id]
        predictions = self.pred_mat[user_id].copy()
        predictions[~user_mask] = -np.inf
        valid_items = user_mask.sum()
        local_k = min(k, valid_items)
        top_k_indices = np.argpartition(predictions, -local_k)[-local_k:]
        top_k_values = predictions[top_k_indices]
        sorted_top_k_indices = top_k_indices[np.argsort(-top_k_values)]
        return [(self._data.private_items[idx], predictions[idx]) for idx in sorted_top_k_indices]

    def get_model_state(self):
        saving_dict = {}
        saving_dict['pred_mat'] = self.pred_mat
        saving_dict['X'] = self.X
        saving_dict['Y'] = self.Y
        saving_dict['C'] = self.C
        return saving_dict

    def set_model_state(self, saving_dict):
        self.pred_mat = saving_dict['pred_mat']
        self.X = saving_dict['X']
        self.Y = saving_dict['Y']
        self.C = saving_dict['C']

    def prepare_predictions(self):
        self.pred_mat = self.X.dot(self.Y.T)

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)