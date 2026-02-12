import torch
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy
from scipy.sparse.linalg import svds
from tqdm import tqdm
from torchdiffeq import odeint

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class BSPM(RecMixin, BaseRecommenderModel):
    r"""
    Blurring-Sharpening Process Models for Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/2211.09324>`_
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 2048, int, None),
            ("_factors", "factors", "k", 256, int, None),
            #("_alpha", "alpha", "alpha", 0.3, float, None),

            ("_idl_solver", "idl_solver", "idl_slv", 'euler', str, None),
            ("_blur_solver", "blur_solver", "blur_slv", 'euler', str, None),
            ("_sharpen_solver", "sharpen_solver", "shr_slv", 'euler', str, None),

            ("_K_idl", "K_idl", "K_idl", 1.0, int, None),
            ("_T_idl", "T_idl", "T_idl", 1.0, float, None),

            ("_K_b", "K_b", "K_b", 1.0, int, None),
            ("_T_b", "T_b", "T_b", 1.0, float, None),

            ("_K_s", "K_s", "K_s", 1.0, int, None),
            ("_T_s", "T_s", "T_s", 1.0, float, None),

            ("_idl_beta", "idl_beta", "idl_beta", 0.3, float, None),

            ("_final_sharpening", "final_sharpening", "final_shr", True, bool, None),
            ("_sharpening_off", "sharpening_off", "shr_off", False, bool, None),
            ("_t_point_combination", "t_point_combination", "t_point_comb", False, bool, None)
        ]

        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        row, col = data.sp_i_train.nonzero()
        self.inter = coo_matrix((np.ones_like(row, dtype=np.float64), (row, col)),
                                 shape=(self._num_users, self._num_items))
        self.inter = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.array([self.inter.row, self.inter.col])),
            values=torch.FloatTensor(self.inter.data),
            size=self.inter.shape, dtype=torch.float
        ).coalesce().to(self.device)

        self.adj_mat = data.sp_i_train.tolil()
        self.d_mat_i, self.d_mat_i_inv, self.vt, self.norm_adj = None, None, None, None

    def train(self):
        adj_mat = self.adj_mat
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = scipy.sparse.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = scipy.sparse.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = scipy.sparse.diags(1 / d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        _, _, self.vt = svds(self.norm_adj, self._factors)
        del d_mat, d_inv, norm_adj, adj_mat

        def scipy_to_torch_sparse_tensor(sparse_mtx):
            sparse_mtx = sparse_mtx.tocoo()
            indices = torch.tensor([sparse_mtx.row, sparse_mtx.col], dtype=torch.long)
            values = torch.tensor(sparse_mtx.data, dtype=torch.float32)
            shape = sparse_mtx.shape
            return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(self.device)

        linear_Filter = self.norm_adj.T @ self.norm_adj
        self.linear_Filter = scipy_to_torch_sparse_tensor(linear_Filter).to_dense().to(self.device)

        left_mat = self.d_mat_i @ self.vt.T
        right_mat = self.vt @ self.d_mat_i_inv
        self.left_mat, self.right_mat = torch.FloatTensor(left_mat).to(self.device), torch.FloatTensor(right_mat).to(self.device)

        self.idl_times = torch.linspace(0, self._T_idl, self._K_idl + 1).float().to(self.device)
        self.blurring_times = torch.linspace(0, self._T_b, self._K_b + 1).float().to(self.device)
        self.sharpening_times = torch.linspace(0, self._T_s, self._K_s + 1).float().to(self.device)

        self.evaluate()

    def sharpenFunction(self, t, r):
        out = torch.mm(r, self.linear_Filter)
        return -out

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        with torch.no_grad():
            for index, offset in enumerate(tqdm(range(0, self._num_users, self._batch_eval))):
                offset_stop = min(offset + self._batch_eval, self._num_users)
                predictions = self.get_users_rating(offset, offset_stop)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_users_rating(self, batch_start, batch_stop):
        batch = torch.arange(batch_start, batch_stop).to(self.device)
        batch_test = self.inter.index_select(dim=0, index=batch)#.to_dense()

        idl_out = torch.mm(batch_test, self.left_mat @ self.right_mat)
        blurred_out = torch.mm(batch_test, self.linear_Filter)

        if self._sharpening_off == False:
            if self._final_sharpening == True:
                sharpened_out = odeint(func=self.sharpenFunction, y0=self._idl_beta * idl_out + blurred_out,
                                       t=self.sharpening_times, method=self._sharpen_solver)
            else:
                sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times,
                                       method=self._sharpen_solver)

        if self._t_point_combination == True:
            if self._sharpening_off == False:
                U_2 = torch.mean(torch.cat([blurred_out.unsqueeze(0), sharpened_out[1:, ...]], axis=0), axis=0)
            else:
                U_2 = blurred_out
                del blurred_out
        else:
            if self._sharpening_off == False:
                U_2 = sharpened_out[-1]
                del sharpened_out
            else:
                U_2 = blurred_out
                del blurred_out

        if self._final_sharpening == True:
            if self._sharpening_off == False:
                ret = U_2
            elif self._sharpening_off == True:
                ret = self._idl_beta * idl_out + U_2
        else:
            ret = self._idl_beta * idl_out + U_2

        return ret

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
        return "BSPM" \
               + f"_{self.get_params_shortcut()}"