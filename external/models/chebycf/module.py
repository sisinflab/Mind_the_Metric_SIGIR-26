import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch
import torch.nn as nn
from torch import spmm


# This function operates on TORCH TENSORS (GPU)
def get_norm_inter_torch(inter: torch.sparse.Tensor):
    user_degree = torch.sparse.sum(inter, dim=1).to_dense()
    item_degree = torch.sparse.sum(inter, dim=0).to_dense()

    user_d_inv_sqrt = torch.pow(user_degree.clamp(min=1), -0.5).flatten()
    user_d_inv_sqrt[torch.isinf(user_d_inv_sqrt)] = 0.

    item_d_inv_sqrt = torch.pow(item_degree.clamp(min=1), -0.5).flatten()
    item_d_inv_sqrt[torch.isinf(item_d_inv_sqrt)] = 0.

    row, col = inter.indices()
    new_values = inter.values() * user_d_inv_sqrt[row] * item_d_inv_sqrt[col]

    return torch.sparse_coo_tensor(inter.indices(), new_values, inter.shape).coalesce()


# --- ADDED: Helper function for SVD that runs on SCIPY/NUMPY (CPU) ---
def get_norm_inter_cpu(inter: sp.spmatrix):
    user_degree = np.array(inter.sum(axis=1)).flatten()
    item_degree = np.array(inter.sum(axis=0)).flatten()
    user_d_inv_sqrt = np.power(user_degree.clip(min=1), -0.5)
    item_d_inv_sqrt = np.power(item_degree.clip(min=1), -0.5)
    user_d_inv_sqrt[user_degree == 0] = 0
    item_d_inv_sqrt[item_degree == 0] = 0
    user_d_inv_sqrt = sp.diags(user_d_inv_sqrt)
    item_d_inv_sqrt = sp.diags(item_d_inv_sqrt)
    norm_inter = (user_d_inv_sqrt @ inter @ item_d_inv_sqrt).tocoo()
    return norm_inter


class Laplacian(nn.Module):
    def __init__(self, inter):
        super().__init__()
        norm_inter = get_norm_inter_torch(inter)
        self.register_buffer('norm_inter', norm_inter)  # shape (num_users, num_items)

    def __mul__(self, x):
        # L_tilde = 2L/lambda_max - I
        # = 2 (I - R_tilde^T * R_tilde)/1 - I
        # = R_tilde^T * R_tilde * -2 + I
        y = spmm(self.norm_inter, x)
        y = spmm(self.norm_inter.t(), y) * (-2)
        y += x
        return y


class ChebyFilter(nn.Module):
    def __init__(self, order, flatness, device):
        super().__init__()
        self.order = order
        self.flatness = flatness
        self.device = device

    def plateau(self):
        x = torch.arange(self.order + 1, device=self.device)
        x = torch.cos((self.order - x) / self.order * math.pi).round(decimals=3)
        output = torch.zeros_like(x)
        output[x < 0] = (-x[x < 0]).pow(self.flatness) * 0.5 + 0.5
        output[x >= 0] = (x[x >= 0]).pow(self.flatness) * (-0.5) + 0.5
        return output.round(decimals=3)

    def cheby(self, x, init):
        if self.order == 0: return [init]
        output = [init, x * init]
        for _ in range(2, self.order + 1):
            output.append(x * output[-1] * 2 - output[-2])
        return torch.stack(output)

    def fit(self, inter):
        # Laplacian_tilde
        self.laplacian = Laplacian(inter)  # shape (num_items, num_items)

        # Chebyshev Nodes and Target Transfer Function Values
        cheby_nodes = torch.arange(1, (self.order + 1) + 1, device=self.device)
        cheby_nodes = torch.cos(((self.order + 1) + 0.5 - cheby_nodes) / (self.order + 1) * math.pi)
        target = self.plateau()
        # Chebyshev Interpolation Coefficients
        coeffs = self.cheby(x=cheby_nodes, init=target).sum(dim=1) * (2 / (self.order + 1))
        coeffs[0] /= 2
        self.register_buffer('coeffs', coeffs)

    def forward(self, signal):
        signal = signal.T
        bases = self.cheby(x=self.laplacian, init=signal)
        output = torch.einsum('K,KNB->BN', self.coeffs, bases)
        return output


class IdealFilter(nn.Module):
    def __init__(self, threshold, weight, device):
        super().__init__()
        self.threshold = threshold
        self.weight = weight
        self.device = device

    def fit(self, inter):
        inter = inter.cpu().coalesce()
        inter = sp.coo_matrix(
            (inter.values().numpy(), (inter.indices()[0].numpy(), inter.indices()[1].numpy())),
            shape=inter.shape
        )
        norm_inter = get_norm_inter_cpu(inter)
        _, _, vt = svds(norm_inter, which='LM', k=self.threshold)
        ideal_pass = torch.tensor(vt.T.copy(), device=self.device, dtype=torch.float32)
        self.register_buffer('ideal_pass', ideal_pass)  # shape (num_items, threshold)

    def forward(self, signal):
        ideal_preds = signal @ self.ideal_pass @ self.ideal_pass.T
        return ideal_preds * self.weight


class DegreeNorm(nn.Module):
    def __init__(self, power, device):
        super().__init__()
        self.power = power
        self.device = device

    def fit(self, inter):
        item_degree = torch.sparse.sum(inter, dim=0).to_dense()
        zero_mask = (item_degree == 0)
        pre_norm = item_degree.clamp(min=1).pow(-self.power)
        pst_norm = item_degree.clamp(min=1).pow(+self.power)
        pre_norm[zero_mask], pst_norm[zero_mask] = 0, 0
        self.register_buffer('pre_normalize', pre_norm)  # (num_items,)
        self.register_buffer('post_normalize', pst_norm)  # (num_items,)

    def forward_pre(self, signal):
        return signal * self.pre_normalize

    def forward_post(self, signal):
        return signal * self.post_normalize