import scipy as sp
from tqdm import tqdm
import numpy as np
import torch
import os
import random
import scipy.sparse as sp
import gc

from elliot.utils.write import store_recommendation
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .GDEModel import GDEModel


class GDE(RecMixin, BaseRecommenderModel):
    r"""
    Less is More: Reweighting Important Spectral Graph Features for Recommendation

    For further details, please refer to the `paper <http://arxiv.org/abs/2204.11346>`_

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        if self._batch_size < 1:
            self._batch_size = self._num_users

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.03, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_beta", "beta", "beta", 5.0, float, None),
            ("_loss_type", "loss_type", "loss_type", "adaptive", str, None),
            ("_feature_type", "feature_type", "feature_type", "both", str, None),
            ("_smooth_ratio", "smooth_ratio", "smooth_ratio", 0.1, float, None),
            ("_rough_ratio", "rough_ratio", "rough_ratio", 0.0, float, None),
            ("_dropout", "dropout", "dropout", 0.1, float, None)
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

        smooth_uu_value, smooth_uu_vector, rough_uu_value, rough_uu_vector, smooth_ii_value, smooth_ii_vector, rough_ii_value, rough_ii_vector = None, None, None, None, None, None, None, None

        if self._config.data_config.strategy == 'fixed':
            self.logger.info(f"Preprocessing {self._config.data_config.train_path}")
            smooth_uu_value, smooth_uu_vector, rough_uu_value, rough_uu_vector, smooth_ii_value, smooth_ii_vector, rough_ii_value, rough_ii_vector = self.preprocess()
            self.logger.info(f"Preprocessing finished!")
        else:
            raise NotImplementedError('The check when strategy is different from fixed has not been implemented yet!')

        self._model = GDEModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            factors=self._factors,
            l_w=self._l_w,
            beta=self._beta,
            dropout=self._dropout,
            loss_type=self._loss_type,
            feature_type=self._feature_type,
            smooth_uu_value=smooth_uu_value,
            smooth_uu_vector=smooth_uu_vector,
            rough_uu_value=rough_uu_value,
            rough_uu_vector=rough_uu_vector,
            smooth_ii_value=smooth_ii_value,
            smooth_ii_vector=smooth_ii_vector,
            rough_ii_value=rough_ii_value,
            rough_ii_vector=rough_ii_vector,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "GDE" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def cal_spectral_feature(self, Adj, k, largest=True, niter=20, oversample=10):
        """
        Adj: sparse symmetric PSD matrix
        k: number of spectral components
        largest=True  -> smooth features
        largest=False -> rough features
        """

        if largest:
            # ===== smooth =====
            U, S, _ = torch.svd_lowrank(
                Adj,
                q=k + oversample,
                niter=niter
            )
            value = S[:k]
            vector = U[:, :k]
            return value, vector

        else:
            # ===== rough =====
            # lambda_max estimation
            U1, S1, _ = torch.svd_lowrank(
                Adj,
                q=1,
                niter=5
            )
            lambda_max = S1[0].detach()

            # build A_shift = lambda_max * I - Adj
            n = Adj.size(0)
            eye_idx = torch.arange(n, device=Adj.device)
            I = torch.sparse_coo_tensor(
                indices=torch.stack([eye_idx, eye_idx]),
                values=torch.full((n,), lambda_max, device=Adj.device),
                size=Adj.size()
            )

            Adj_shift = I - Adj

            # SVD on top of Adj_shift
            U, S, _ = torch.svd_lowrank(
                Adj_shift,
                q=k + oversample,
                niter=niter
            )

            # original small eigenvalues
            value = lambda_max - S[:k]
            vector = U[:, :k]

            return value, vector

    def preprocess(self):
        D_u = torch.sparse.sum(self.rate_matrix, dim=1).to_dense()
        D_i = torch.sparse.sum(self.rate_matrix, dim=0).to_dense()
        for i in range(self._num_users):
            if D_u[i] != 0:
                D_u[i] = 1 / D_u[i].sqrt()
        for i in range(self._num_items):
            if D_i[i] != 0:
                D_i[i] = 1 / D_i[i].sqrt()
        rate_matrix = D_u.unsqueeze(1) * self.rate_matrix * D_i

        del D_u, D_i
        gc.collect()
        torch.cuda.empty_cache()

        # user-user relations
        L_u = torch.sparse.mm(rate_matrix, rate_matrix.t())
        smooth_uu_value, smooth_uu_vector = self.cal_spectral_feature(L_u,
                                                                      int(self._smooth_ratio * self._num_users),
                                                                      largest=True)
        gc.collect()
        torch.cuda.empty_cache()
        if self._rough_ratio != 0:
            rough_uu_value, rough_uu_vector = self.cal_spectral_feature(L_u,
                                                                        int(self._rough_ratio * self._num_users),
                                                                        largest=False)
        else:
            rough_uu_value, rough_uu_vector = None, None

        del L_u
        gc.collect()
        torch.cuda.empty_cache()

        # item-item relations
        L_i = torch.sparse.mm(rate_matrix.t(), rate_matrix)
        smooth_ii_value, smooth_ii_vector = self.cal_spectral_feature(L_i,
                                                                      int(self._smooth_ratio * self._num_items),
                                                                      largest=True)
        gc.collect()
        torch.cuda.empty_cache()
        if self._rough_ratio != 0:
            rough_ii_value, rough_ii_vector = self.cal_spectral_feature(L_i,
                                                                        int(self._rough_ratio * self._num_items),
                                                                        largest=False)
        else:
            rough_ii_value, rough_ii_vector = None, None

        del L_i
        gc.collect()
        torch.cuda.empty_cache()

        return smooth_uu_value, smooth_uu_vector, rough_uu_value, rough_uu_vector, smooth_ii_value, smooth_ii_vector, rough_ii_value, rough_ii_vector

    def train(self):
        if self._restore:
            return self.restore_weights()

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
        final_user_embeddings = self._model.L_u.mm(self._model.user_embed.weight)
        final_item_embeddings = self._model.L_i.mm(self._model.item_embed.weight)
        batch_user_embeddings = final_user_embeddings[batch_start:batch_stop]
        predictions = (batch_user_embeddings.mm(final_item_embeddings.t())).sigmoid()
        return predictions

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))
