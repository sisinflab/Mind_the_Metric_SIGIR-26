import torch
from tqdm import tqdm

def batch_dense_matmul(A, B, device="cuda", batch_size=1000):
    """
    Calcola A @ B a blocchi, scrivendo i risultati direttamente
    in una matrice preallocata.

    Args:
        A (torch.Tensor): Primo tensore denso, shape (n, k).
        B (torch.Tensor): Secondo tensore denso, shape (k, m).
        device (str): Device di calcolo ("cpu" o "cuda").
        batch_size (int): Dimensione del batch per l'elaborazione.

    Returns:
        torch.Tensor: Risultato della moltiplicazione come tensore denso, shape (n, m).
    """

    n, k = A.shape
    k2, m = B.shape
    assert k == k2, f"Dimensioni incompatibili: A={A.shape}, B={B.shape}"

    result = torch.empty((n, m), dtype=A.dtype, device=device)

    for start in tqdm(range(0, n, batch_size), disable=False):
        end = min(start + batch_size, n)
        result[start:end, :] = torch.matmul(A[start:end, :], B)
    return result