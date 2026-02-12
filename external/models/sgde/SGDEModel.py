from abc import ABC

import torch
import numpy as np
import random


class SGDEModel(torch.nn.Module, ABC):
    def __init__(self,
                 learning_rate,
                 embed_k,
                 l_w,
                 beta,
                 req_vec,
                 u,
                 value,
                 v,
                 random_seed,
                 **kwargs
                 ):
        super().__init__()

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.beta = beta

        svd_filter = self.weight_func(value[:req_vec].to(self.device))
        self.user_vector = (u[:, :req_vec]).to(self.device) * svd_filter
        self.item_vector = (v[:, :req_vec]).to(self.device) * svd_filter
        self.FS = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.randn(req_vec, self.embed_k), -np.sqrt(6. / (req_vec + self.embed_k)),
                                   np.sqrt(6. / (req_vec + self.embed_k))).to(self.device))

    def weight_func(self, sig):
        return torch.exp(self.beta * sig)

    def train_step(self, batch):
        u, p, n = batch

        final_user = self.user_vector[u].mm(self.FS)
        final_pos = self.item_vector[p].mm(self.FS)
        final_nega = self.item_vector[n].mm(self.FS)

        out = ((final_user * final_pos).sum(1) - (final_user * final_nega).sum(1)).sigmoid()

        regu_term = self.l_w * (final_user ** 2 + final_pos ** 2 + final_nega ** 2).sum()

        loss = (-torch.log(out + 1e-8).sum() + regu_term)/u.shape[0]  # fixed
        loss.backward()
        with torch.no_grad():
            self.FS -= self.learning_rate * self.FS.grad
            self.FS.grad.zero_()

        return loss.item()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
