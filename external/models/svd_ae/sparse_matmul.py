import torch
from tqdm import tqdm


def batch_dense_matmul_sparse_input(A, B, device, batch_size=1000):
    """
    Calculates A @ B in blocks, where A is dense and B is sparse.
    It returns a dense tensor as output.

    This function is useful when the matrix A is very large and its
    multiplication with B could cause memory problems on the GPU,
    while the final result can be handled in memory.

    Args:
        A (torch.Tensor): First tensor (dense).
        B (torch.sparse_coo_tensor): Second tensor (sparse).
        device (torch.device or str): (es. 'cuda' o 'cpu')
        batch_size (int): Batch size for processing rows of A.

    Returns:
        torch.Tensor: Multiplication result as a dense tensor.
    """
    # Checking dimensional compatibility for proper executability of multiplication
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"The dimensions of the matrices are not compatible for multiplication: A.shape={A.shape}, B.shape={B.shape}")

    B = B.coalesce().to(device)

    n, m = A.shape[0], B.shape[1]

    # List of partial results
    result_rows = []

    for offset in tqdm(range(0, n, batch_size), desc="Batch MatMul (Dense @ Sparse)"):
        offset_stop = min(offset + batch_size, n)
        A_block = A[offset:offset_stop, :].to(device) # extracting batch
        # Dense @ Sparse -> Dense
        block_result = torch.matmul(A_block, B)

        # Moves partial result to CPU to free GPU memory
        result_rows.append(block_result.cpu())
        # Explicit GPU memory cleaning
        del A_block, block_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del B
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Concatenates all result blocks on CPU
    return torch.cat(result_rows, dim=0).to(device)
