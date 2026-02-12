from abc import ABC

import torch
import numpy as np
import random


class CSGDEModel(torch.nn.Module, ABC):
    def __init__(self,
                 learning_rate,
                 embed_k,
                 l_w,
                 beta,
                 req_vec,
                 std,
                 coef_u,
                 coef_i,
                 u,
                 value,
                 v,
                 random_seed,
                 name="SVDGCN",
                 **kwargs
                 ):
        super().__init__()

        # set seed
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
        self.std = std
        self.beta = beta
        self.coef_u = coef_u
        self.coef_i = coef_i

        svd_filter = self.weight_func(value[:req_vec].to(self.device))
        self.user_vector = (u[:, :req_vec]).to(self.device) * svd_filter
        self.item_vector = (v[:, :req_vec]).to(self.device) * svd_filter
        self.FS = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.randn(req_vec, self.embed_k), -np.sqrt(6. / (req_vec + self.embed_k)),
                                   np.sqrt(6. / (req_vec + self.embed_k))).to(self.device))

    def weight_func(self, sig):
        return torch.exp(self.beta * sig)

    # def forward(self, inputs, **kwargs):
    #     emb1, emb2 = inputs
    #     emb1_final = torch.squeeze(emb1).to(self.device).mm(self.FS)
    #     emb2_final = torch.squeeze(emb2).to(self.device).mm(self.FS)
    #
    #     out = torch.sum(emb1_final * emb2_final, 1)
    #
    #     return emb1_final, emb2_final, out

    def train_step(self, batch):
        u, p, n, up, un, pp, pn = batch

        final_user = torch.normal(self.user_vector[u],std=self.std).mm(self.FS)
        final_pos = torch.normal(self.item_vector[p],std=self.std).mm(self.FS)
        final_nega = torch.normal(self.item_vector[n],std=self.std).mm(self.FS)

        final_user_p = torch.normal(self.user_vector[up], std=self.std).mm(self.FS)
        final_user_n = torch.normal(self.user_vector[un], std=self.std).mm(self.FS)

        final_pos_p = torch.normal(self.item_vector[pp], std=self.std).mm(self.FS)
        final_pos_n = torch.normal(self.item_vector[pn], std=self.std).mm(self.FS)

        out = ((final_user * final_pos).sum(1) - (final_user * final_nega).sum(1)).sigmoid()

        self_loss_u = torch.log(((final_user * final_user_p).sum(1) - (final_user * final_user_n).sum(1)).sigmoid()).sum()
        self_loss_i = torch.log(((final_pos * final_pos_p).sum(1) - (final_pos * final_pos_n).sum(1)).sigmoid()).sum()

        regu_term = self.l_w *(final_user**2+final_pos**2+final_nega**2+final_user_p**2+final_user_n**2+final_pos_p**2+final_pos_n**2).sum()

        loss = (-torch.log(out + 1e-8).sum()-self.coef_u*self_loss_u-self.coef_i*self_loss_i+regu_term)/u.shape[0]  # fixed
        loss.backward()
        with torch.no_grad():
            self.FS -= self.learning_rate * self.FS.grad
            self.FS.grad.zero_()

        return loss.item() #.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
