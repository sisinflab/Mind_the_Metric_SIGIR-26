from scipy.sparse import coo_matrix
from tqdm import tqdm
import time

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


from .module import *


class ChebyCF(RecMixin, BaseRecommenderModel):
    r"""
    Graph Spectral Filtering with Chebyshev Interpolation for Recommendation

    For further details, please refer to the `paper <http://arxiv.org/abs/2505.00552>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 1024, int, None),
            ("_K", "K", "K", 8, int, None),
            ("_phi", "phi", "phi", 10, float, None),
            ("_eta", "eta", "eta", 256, int, None),
            ("_alpha", "alpha", "alpha", 0.1, float, None),
            ("_beta", "beta", "beta", 0.1, float, None)
        ]

        self.autoset_params()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Running on: {self.device}")

        self.cheby = ChebyFilter(self._K, self._phi, self.device)
        self.ideal = IdealFilter(self._eta, self._alpha, self.device) if self._eta > 0 and self._alpha > 0 else None
        self.norm = DegreeNorm(self._beta, self.device) if self._beta > 0 else None

        # row, col = data.sp_i_train.nonzero()
        # self.inter = coo_matrix((np.ones_like(row, dtype=np.float32), (row, col)),
        #                         shape=(self._num_users, self._num_items)).to(self.device)

        sp_i_train = data.sp_i_train.tocoo()
        indices = torch.LongTensor(np.vstack([sp_i_train.row, sp_i_train.col]))
        values = torch.FloatTensor(sp_i_train.data)
        self.inter = torch.sparse_coo_tensor(
            indices, values, sp_i_train.shape,
            dtype=torch.float32
        ).coalesce().to(self.device)

    def train(self):
        start = time.time()

        self.cheby.fit(self.inter)
        if self.ideal:
            self.ideal.fit(self.inter)
        if self.norm:
            self.norm.fit(self.inter)

        end = time.time()
        self.logger.info(f"Training time: {end - start} seconds")

        # self.inter = torch.sparse_coo_tensor(
        #     indices=torch.LongTensor(np.array([self.inter.row, self.inter.col])),
        #     values=torch.FloatTensor(self.inter.data),
        #     size=self.inter.shape
        # ).coalesce()

        self.evaluate()

    def get_users_rating(self, batch_start, batch_stop):
        batch = torch.arange(batch_start, batch_stop, device=self.device)
        signal = self.inter.index_select(dim=0, index=batch).to_dense()
        if self.norm:
            signal = self.norm.forward_pre(signal)
        output = self.cheby.forward(signal)
        if self.ideal:
            output += self.ideal.forward(signal)
        if self.norm:
            output = self.norm.forward_post(output)
        return output

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
        return "ChebyCF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"