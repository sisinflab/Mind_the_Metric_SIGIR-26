import torch
import numpy as np
from scipy.sparse import coo_matrix, diags, hstack
from scipy.sparse.linalg import LinearOperator, eigsh
import time
from tqdm import tqdm

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class PGSP(RecMixin, BaseRecommenderModel):
    r"""
    Personalized Graph Signal Processing for Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/2302.02113>`_
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
            ("_factors", "factors", "factors", 512, int, None),
            ("_phi", "phi", "phi", 0.7, float, None),
            ("_P0_", "P0_", "P0_", True, bool, None),
            ("_P1_", "P1_", "P1_", True, bool, None)
        ]

        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        start = time.time()

        row, col = self._data.sp_i_train.nonzero()
        R = coo_matrix((np.ones_like(row, dtype=np.float64), (row, col)), shape=(self._num_users, self._num_items))

        Du_ = diags(np.power(R.sum(axis=1).T.A[0], -1/2), offsets=0)
        Di_ = diags(np.power(R.sum(axis=0).A[0], -1/2), offsets=0)
        # Du  = diags(np.power(R.sum(axis=1).T.A[0], 1/2), offsets=0)
        Di  = diags(np.power(R.sum(axis=0).A[0], 1/2), offsets=0)
        Ru = Du_ * R
        Ri = R * Di_
        # Cu = Ri * Ri.T
        # Ci = Ru.T * Ru
        R_post = Ru * Di_
        Ci0 = R_post.T * R_post
        Cu0 = R_post * R_post.T
        del R_post, row, col

        def A_matvec(x):
            """
            x: vector length N_total (first users, then items)
            returns A.dot(x)
            Uses sparse multiplications and avoids forming Cu and Ci explicitly.
            """
            u = x[:self._num_users].astype(np.float64)
            i = x[self._num_users:].astype(np.float64)

            # Cu * u = Ri * (Ri.T * u)
            tmp = Ri.T.dot(u)  # shape (items,)
            Cu_u = Ri.dot(tmp)  # shape (users,)

            # Ci * i = Ru.T * (Ru * i)
            tmp2 = Ru.dot(i)  # shape (users,)
            Ci_i = Ru.T.dot(tmp2)  # shape (items,)

            # R * i and R.T * u
            R_i = R.dot(i)  # users
            RT_u = R.T.dot(u)  # items

            top = Cu_u + R_i
            bottom = RT_u + Ci_i
            return np.concatenate([top, bottom]).astype(np.float64)

        N_total = self._num_users + self._num_items

        ones = np.ones(N_total, dtype=np.float64)
        deg = A_matvec(ones)  # shape N_total

        # prevent zeros:
        deg[deg <= 0] = 1.0
        D_inv_sqrt = np.power(deg, -0.5).astype(np.float32)  # length N_total

        # LinearOperator for A_norm = D_ * A * D_
        def A_norm_matvec(x):
            # x length N_total
            y = A_matvec((D_inv_sqrt * x).astype(np.float64))
            return (D_inv_sqrt * y).astype(np.float64)

        # L_norm operator: I - A_norm
        def L_norm_matvec(x):
            return x - A_norm_matvec(x)

        # A_linop = LinearOperator(shape=(N_total, N_total), matvec=A_norm_matvec, dtype=np.float64)
        L_linop = LinearOperator(shape=(N_total, N_total), matvec=L_norm_matvec, dtype=np.float64)

        _, vec = eigsh(L_linop, k=self._factors, which='SA')
        del L_linop

        R_b = hstack([Cu0, R])
        D_Rb_i_ = diags(np.power(R_b.sum(axis=0).A[0], -1/2), offsets=0)
        D_Rb_i = diags(np.power(R_b.sum(axis=0).A[0], 1/2), offsets=0)
        D_Rb_i = D_Rb_i.toarray()

        if self._P0_:
            P0 = R * Di_
            P0 = P0 * Ci0
            P0 = P0 * Di
            P0 = P0.toarray()
        else:
            P0 = R * Ci0
            P0 = P0.toarray()
        del R, Di_, Ci0, Di

        if self._P1_:
            P1 = R_b * D_Rb_i_
            P1 = P1 * vec
            P11 = np.matmul(vec.T, D_Rb_i)
            P1 = np.matmul(P1, P11)
            P1 = P1[:, self._num_users:]
        else:
            P1 = R_b * vec
            P1 = np.matmul(P1, vec.T)
            P1 = P1[:, self._num_users:]
        del R_b, D_Rb_i_, vec, D_Rb_i

        self._preds = self._phi * P0 + (1 - self._phi) * P1

        end = time.time()
        self.logger.info(f"Training has taken: {end-start}")

        self.evaluate()

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

    def get_users_rating(self, batch_start, batch_stop):
        return self._preds[batch_start:batch_stop, :]

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
        return "PGSP" \
               + f"_{self.get_params_shortcut()}"