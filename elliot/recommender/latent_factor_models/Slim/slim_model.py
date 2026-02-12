"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import time
import numpy as np
import sys
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import multiprocessing
from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
from tqdm import tqdm

# worker function that runs in separate processes
@ignore_warnings(category=ConvergenceWarning)
def _train_slim_chunk(items_to_process,
                      shape,
                      neighborhood,
                      alpha,
                      l1_ratio,
                      random_seed,
                      shm_names,
                      shm_shapes,
                      shm_dtypes):
    indptr_shm = shared_memory.SharedMemory(name=shm_names['indptr'], create=False)
    indices_shm = shared_memory.SharedMemory(name=shm_names['indices'], create=False)
    data_shm = shared_memory.SharedMemory(name=shm_names['data'], create=False)

    train = sp.csc_matrix((
        np.ndarray(shm_shapes['data'], dtype=shm_dtypes['data'], buffer=data_shm.buf),
        np.ndarray(shm_shapes['indices'], dtype=shm_dtypes['indices'], buffer=indices_shm.buf),
        np.ndarray(shm_shapes['indptr'], dtype=shm_dtypes['indptr'], buffer=indptr_shm.buf),
    ), shape=shape)

    rows, cols, values = [], [], []

    # Loop over items assigned to this worker
    for currentItem in items_to_process:
        md = ElasticNet(alpha=alpha,
                        l1_ratio=l1_ratio,
                        positive=True,
                        fit_intercept=False,
                        copy_X=False,
                        precompute=True,
                        selection='random',
                        max_iter=100,
                        random_state=random_seed + currentItem,
                        tol=1e-4)

        y = train[:, currentItem].toarray()

        start_pos = train.indptr[currentItem]
        end_pos = train.indptr[currentItem + 1]
        current_item_data_backup = train.data[start_pos: end_pos].copy()
        train.data[start_pos: end_pos] = 0.0

        md.fit(train, y)

        nonzero_model_coef_index = md.sparse_coef_.indices
        nonzero_model_coef_value = md.sparse_coef_.data

        local_topK = min(len(nonzero_model_coef_value) - 1, neighborhood)

        if local_topK < 0:
            train.data[start_pos:end_pos] = current_item_data_backup
            continue

        relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
        relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        for index in range(len(ranking)):
            rows.append(nonzero_model_coef_index[ranking[index]])
            cols.append(currentItem)
            values.append(nonzero_model_coef_value[ranking[index]])

        train.data[start_pos:end_pos] = current_item_data_backup

    indptr_shm.close()
    indices_shm.close()
    data_shm.close()

    return values, rows, cols


class SlimModel(object):
    def __init__(self,
                 data, num_users, num_items, l1_ratio, alpha, epochs, neighborhood, random_seed):

        self._data = data
        self._num_users = num_users
        self._num_items = num_items
        self._l1_ratio = l1_ratio
        self._alpha = alpha
        self._epochs = epochs
        self._neighborhood = neighborhood
        self._random_seed = random_seed # Memorizza il random seed

        self.md = ElasticNet(alpha=self._alpha,
                             l1_ratio=self._l1_ratio,
                             positive=True,
                             fit_intercept=False,
                             copy_X=False,
                             precompute=True,
                             selection='random',
                             max_iter=100,
                             random_state=random_seed,
                             tol=1e-4)

        self._w_sparse = None
        self.pred_mat = None

    def train(self, verbose=True, workers=cpu_count()):
        """
        Train the SLIM model using parallel processes.

        Args:
            verbose (bool): if True, displays a progress bar.
            workers (int): Number of worker processes to use. By default, use all CPU cores.
        """

        train = self._data.sp_i_train_ratings.tocsc()

        # 1. Creating shared memory blocks for sparse matrix data
        shm_indptr = shared_memory.SharedMemory(create=True, size=train.indptr.nbytes)
        shm_indices = shared_memory.SharedMemory(create=True, size=train.indices.nbytes)
        shm_data = shared_memory.SharedMemory(create=True, size=train.data.nbytes)

        # Copy the matrix data to the shared memory buffers
        b_indptr = np.ndarray(train.indptr.shape, dtype=train.indptr.dtype, buffer=shm_indptr.buf)
        b_indptr[:] = train.indptr[:]
        b_indices = np.ndarray(train.indices.shape, dtype=train.indices.dtype, buffer=shm_indices.buf)
        b_indices[:] = train.indices[:]
        b_data = np.ndarray(train.data.shape, dtype=train.data.dtype, buffer=shm_data.buf)
        b_data[:] = train.data[:]

        shm_names = {
            'indptr': shm_indptr.name,
            'indices': shm_indices.name,
            'data': shm_data.name
        }
        shm_shapes = {
            'indptr': train.indptr.shape,
            'indices': train.indices.shape,
            'data': train.data.shape
        }
        shm_dtypes = {
            'indptr': train.indptr.dtype,
            'indices': train.indices.dtype,
            'data': train.data.dtype
        }

        # 2. Prepare the partial function and item chunks
        _pfit = partial(_train_slim_chunk,
                        shape=train.shape,
                        neighborhood=self._neighborhood,
                        alpha=self._alpha,
                        l1_ratio=self._l1_ratio,
                        random_seed=self._random_seed,
                        shm_names=shm_names,
                        shm_shapes=shm_shapes,
                        shm_dtypes=shm_dtypes)

        item_chunksize = max(1, int(self._num_items / (workers * 4)))
        item_chunks = np.array_split(np.arange(self._num_items), int(self._num_items / item_chunksize))

        all_values, all_rows, all_cols = [], [], []

        # 3. Run the process pool
        start_time = time.time()
        print(f"Avvio del training parallelo di SLIM con {workers} worker...")

        with Pool(processes=workers) as pool:
            # imap_unordered is efficient because it processes results as soon as they are ready.
            results_iterator = pool.imap_unordered(_pfit, item_chunks)

            if verbose:
                results_iterator = tqdm(results_iterator, total=len(item_chunks), desc="Training Chunks")

            for values_chunk, rows_chunk, cols_chunk in results_iterator:
                all_values.extend(values_chunk)
                all_rows.extend(rows_chunk)
                all_cols.extend(cols_chunk)

        # 4. Clean shared memory
        shm_indptr.close()
        shm_indices.close()
        shm_data.close()

        try:
            shm_indptr.unlink()
            shm_indices.unlink()
            shm_data.unlink()
        except FileNotFoundError:
            pass

        # 5. Build final sparse weight matrix
        self._w_sparse = sp.csr_matrix((all_values, (all_rows, all_cols)),
                                      shape=(self._num_items, self._num_items), dtype=np.float32)

    def prepare_predictions(self):
        self.pred_mat = self._data.sp_i_train_ratings.dot(self._w_sparse).toarray()

    def predict(self, u, i):
        return self.pred_mat[u, i]

    def get_user_recs(self, user, mask, k=100):
        ui = self._data.public_users[user]
        user_mask = mask[ui]
        predictions = self.pred_mat[ui].copy()
        predictions[~user_mask] = -np.inf
        valid_items = user_mask.sum()
        local_k = min(k, valid_items)
        # Gestisce il caso in cui non ci siano item validi
        if local_k == 0:
            return []
        top_k_indices = np.argpartition(predictions, -local_k)[-local_k:]
        top_k_values = predictions[top_k_indices]
        sorted_top_k_indices = top_k_indices[np.argsort(-top_k_values)]
        return [(self._data.private_items[idx], predictions[idx]) for idx in sorted_top_k_indices]