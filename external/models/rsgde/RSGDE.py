import scipy as sp
from tqdm import tqdm
import numpy as np
import torch
import os
import random
import scipy.sparse as sp

from elliot.utils.write import store_recommendation
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .RSGDEModel import RSGDEModel


class RSGDE(RecMixin, BaseRecommenderModel):
    r"""
    RSGDE: Robust Simplified Graph Denoising Encoder

    from the article "Less is More: Removing Redundancy of Graph Convolutional Networks for Recommendation"

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3632751>`_
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 7.5, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_alpha", "alpha", "alpha", 2, float, None),
            ("_beta", "beta", "beta", 0.2, float, None),
            ("_req_vec", "req_vec", "req_vec", 60, int, None),
            ("_std", "std", "std", 0.01, float, None),
        ]
        self.autoset_params()

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        row, col = data.sp_i_train.nonzero()
        self.rate_matrix = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.array([row, col])),
            values=torch.FloatTensor(np.ones_like(row)),
            size=(self._num_users, self._num_items), dtype=torch.float
        ).coalesce().to(self.device)

        if self._config.data_config.strategy == 'fixed':
            path = os.path.split(self._config.data_config.train_path)[0]
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

        self._model = RSGDEModel(
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            beta=self._beta,
            req_vec=self._req_vec,
            std=self._std,
            u=U,
            value=value,
            v=V,
            random_seed=self._seed
        )

        del U, value, V
        torch.cuda.empty_cache()

    @property
    def name(self):
        return "RSGDE" \
               + f"_{self.get_base_params_shortcut()}" \
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
        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for _, _ in enumerate(range(0, self._data.transactions, self._batch_size)):
                    steps += 1

                    u = torch.from_numpy(np.random.randint(0, self._num_users, self._batch_size)).to(self.device)

                    rate_dense = self.rate_matrix.index_select(dim=0, index=u).to_dense()

                    p = torch.multinomial(rate_dense, 1, True).squeeze(1)
                    nega = torch.multinomial(1 - rate_dense, 1, True).squeeze(1)

                    batch = u, p, nega
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

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
        final_user = self._model.user_vector[batch_start: batch_stop].mm(self._model.FS)
        final_item = self._model.item_vector.mm(self._model.FS).to(self.device)
        return (final_user.mm(final_item.t())).sigmoid().to(self.device)

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))
