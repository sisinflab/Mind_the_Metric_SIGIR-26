import numpy as np
import torch
import os

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .SVDGCNSModel import SVDGCNSModel


class SVDGCNS(RecMixin, BaseRecommenderModel):
    r"""
    SVD-GCN-S: A Simplified Graph Convolution Paradigm for Recommendation
    (Non-trainable version)

    Args:
        batch_size: Batch size
        beta: Beta parameter for weighting function
        req_vec: No description
        alpha: For the R normalization

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        if self._batch_size < 1:
            self._batch_size = self._num_users

        self._params_list = [
            ("_alpha", "alpha", "alpha", 1, float, None),
            ("_beta", "beta", "beta", 0.1, float, None),
            ("_req_vec", "req_vec", "req_vec", 90, int, None)
        ]
        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        row, col = data.sp_i_train.nonzero()
        self.rate_matrix = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.array([row, col])),
            values=torch.FloatTensor(np.ones_like(row)),
            size=(self._num_users, self._num_items), dtype=torch.float
        ).coalesce().to(self.device)

        if self._config.data_config.strategy == 'fixed':
            path, _ = os.path.split(self._config.data_config.train_path)
            if not (os.path.exists(path + f'/svd_u_{self._alpha}.npy')
                    or os.path.exists(path + f'/svd_i_{self._alpha}.npy') or os.path.exists(
                    path + f'/svd_value_{self._alpha}.npy')):
                self.logger.info(
                    f"Processing singular values as they haven't been calculated before on this dataset...")
                U, value, V = self.preprocess(path, data.num_users, data.num_items)
                self.logger.info(f"Processing end!")
            else:
                self.logger.info(f"Singular values have already been processed for this dataset!")
                value = torch.Tensor(np.load(path + f'/svd_value_{self._alpha}.npy'))
                U = torch.Tensor(np.load(path + f'/svd_u_{self._alpha}.npy'))
                V = torch.Tensor(np.load(path + f'/svd_v_{self._alpha}.npy'))
        else:
            raise NotImplementedError('The check when strategy is different from fixed has not been implemented yet!')

        self._model = SVDGCNSModel(
            req_vec=self._req_vec,
            u=U,
            value=value,
            v=V,
            beta=self._beta
        )

        del U, value, V
        torch.cuda.empty_cache()

    @property
    def name(self):
        return "SVDGCNS" \
               + f"_{self.get_params_shortcut()}"

    def preprocess(self, dataset, users, items):
        D_u = torch.sparse.sum(self.rate_matrix, dim=1).to_dense() + self._alpha
        D_i = torch.sparse.sum(self.rate_matrix, dim=0).to_dense() + self._alpha

        for i in range(users):
            if D_u[i] != 0:
                D_u[i] = 1 / D_u[i].sqrt()

        for i in range(items):
            if D_i[i] != 0:
                D_i[i] = 1 / D_i[i].sqrt()

        # \tilde{R}
        rate_matrix = D_u.unsqueeze(1) * self.rate_matrix * D_i

        # free space
        del D_u, D_i

        U, value, V = torch.svd_lowrank(rate_matrix, q=400, niter=30)

        np.save(dataset + f'/svd_u_{self._alpha}.npy', U.cpu().numpy())
        np.save(dataset + f'/svd_v_{self._alpha}.npy', V.cpu().numpy())
        np.save(dataset + f'/svd_value_{self._alpha}.npy', value.cpu().numpy())

        return U, value, V

    def train(self):
        self.evaluate()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self.get_users_rating(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_users_rating(self, batch_start, batch_stop):
        final_user = self._model.user_vector[batch_start: batch_stop]
        final_item = self._model.item_vector
        return (final_user.mm(final_item.t())).sigmoid().to(self.device)

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))
