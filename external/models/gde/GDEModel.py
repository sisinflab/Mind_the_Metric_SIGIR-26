from abc import ABC

import torch
import numpy as np
import random
import gc


class GDEModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 factors,
                 l_w,
                 beta,
                 dropout,
                 loss_type,
                 feature_type,
                 smooth_uu_value,
                 smooth_uu_vector,
                 rough_uu_value,
                 rough_uu_vector,
                 smooth_ii_value,
                 smooth_ii_vector,
                 rough_ii_value,
                 rough_ii_vector,
                 random_seed,
                 name="GDE",
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

        self.factors = factors
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.loss_type = loss_type
        self.beta = beta
        self.drop_out = dropout
        if self.drop_out != 0:
            self.m = torch.nn.Dropout(self.drop_out)

        self.user_embed = torch.nn.Embedding(num_users, self.factors)
        self.item_embed = torch.nn.Embedding(num_items, self.factors)
        torch.nn.init.xavier_normal_(self.user_embed.weight)
        torch.nn.init.xavier_normal_(self.item_embed.weight)
        self.user_embed.to(self.device)
        self.item_embed.to(self.device)

        if feature_type == 'smoothed':
            user_filter = self.weight_feature(smooth_uu_value)
            item_filter = self.weight_feature(smooth_ii_value)
            user_vector = smooth_uu_vector
            item_vector = smooth_ii_vector
        elif feature_type == 'both':
            user_filter = torch.cat([self.weight_feature(smooth_uu_value), self.weight_feature(rough_uu_value)])
            item_filter = torch.cat([self.weight_feature(smooth_ii_value), self.weight_feature(rough_ii_value)])
            user_vector = torch.cat([smooth_uu_vector, rough_uu_vector], 1)
            item_vector = torch.cat([smooth_ii_vector, rough_ii_vector], 1)
        else:
            raise ValueError("Unknown feature type")

        self.L_u = (user_vector * user_filter).mm(user_vector.t())
        self.L_i = (item_vector * item_filter).mm(item_vector.t())

        del user_vector, item_vector, user_filter, item_filter
        gc.collect()
        torch.cuda.empty_cache()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

    def weight_feature(self, value):
        return torch.exp(self.beta * value)

    def train_step(self, batch):
        user, pos_item, nega_item = batch
        batch_size = user.shape[0]

        if self.drop_out == 0:
            final_user = self.L_u[user].mm(self.user_embed.weight)
            final_pos  = self.L_i[pos_item].mm(self.item_embed.weight)
            final_nega = self.L_i[nega_item].mm(self.item_embed.weight)
        else:
            final_user = (self.m(self.L_u[user]) * (1 - self.drop_out)).mm(self.user_embed.weight)
            final_pos  = (self.m(self.L_i[pos_item]) * (1 - self.drop_out)).mm(self.item_embed.weight)
            final_nega = (self.m(self.L_i[nega_item]) * (1 - self.drop_out)).mm(self.item_embed.weight)

        if self.loss_type == 'adaptive':
            res_nega = (final_user * final_nega).sum(1)
            nega_weight = (1 - (1 - res_nega.sigmoid().clamp(max=0.99)).log10()).detach()
            out = ((final_user * final_pos).sum(1) - nega_weight * res_nega).sigmoid()
        elif self.loss_type == 'bpr':
            out = ((final_user * final_pos).sum(1) - (final_user * final_nega).sum(1)).sigmoid()
        else:
            raise ValueError("Unknown loss type")

        reg_term = self.l_w * (final_user ** 2 + final_pos ** 2 + final_nega ** 2).sum()

        # loss = (-torch.log(out).sum() + reg_term) #/ batch_size
        loss = (-torch.log(out + 1e-8).sum() + reg_term) / batch_size
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
