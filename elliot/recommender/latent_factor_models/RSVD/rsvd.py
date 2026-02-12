"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time
from tqdm import tqdm

from elliot.recommender.recommender_utils_mixin import RecMixin

from .rsvd_model import RSVDModel
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger


class RSVD(RecMixin, BaseRecommenderModel):
    r"""
    RSVD: Regularized Singular Value Decomposition and Application to Recommender System

    For further details, please refer to the `paper <https://arxiv.org/abs/1804.05090>`_

    Args:
        factors: Number of latent factors
        reg: Regularization constant
        seed: Random seed

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        RSVD:
          meta:
            save_recs: True
          factors: 10
          seed: 42
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "factors", 10, int, None),
            ("_reg", "reg", "reg", 10, float, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._model = RSVDModel(self._factors, self._reg, self._data, self._seed)

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        recs = {}
        for i in tqdm(range(0, len(self._ratings.keys()), 1024), desc="Processing batches",
                      total=len(self._ratings.keys()) // 1024 + (1 if len(self._ratings.keys()) % 1024 != 0 else 0)):
            batch = list(self._ratings.keys())[i:i + 1024]
            mat = self._model.get_user_recs(batch, mask, k)
            proc_batch = dict(zip(batch, mat))
            recs.update(proc_batch)
        return recs

    @property
    def name(self):
        return f"RSVD_{self.get_params_shortcut()}"

    def train(self):
        start = time.time()

        self._model.train()

        end = time.time()
        self.logger.info(f"The similarity computation has taken: {end - start}")

        self.evaluate()