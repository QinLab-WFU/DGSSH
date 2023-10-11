import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

def similarityloss(u, target,num_class):
    theta = torch.einsum('ij,jk->ik', u, u.t()) / 2
    one_hot = torch.nn.functional.one_hot(target,num_class)
    one_hot = one_hot.float()
    Sim = (torch.einsum('ij,jk->ik', one_hot, one_hot.t()) > 0).float()
    pair_loss = (torch.log(1 + torch.exp(theta)) - Sim * theta)
    mask_positive = Sim > 0
    mask_negative = Sim <= 0
    S1 = mask_positive.float().sum() - u.shape[0]
    S0 = mask_negative.float().sum()
    if S0 == 0:
        S0 = 1
    if S1 == 0:
        S1 = 1
    S = S0 + S1
    pair_loss[mask_positive] = pair_loss[mask_positive] * (S / S1)
    pair_loss[mask_negative] = pair_loss[mask_negative] * (S / S0)

    diag_matrix = torch.tensor(np.diag(torch.diag(pair_loss.detach()).cpu())).cuda()
    pair_loss = pair_loss - diag_matrix
    count = (u.shape[0] * (u.shape[0] - 1) / 2)

    return pair_loss.sum() / 2 / count



class AsyLoss(nn.Module):
    def __init__(self, beta, code_length, num_train):
        super(AsyLoss, self).__init__()
        self.beta = beta
        self.code_length = code_length
        self.num_train = num_train

    def forward(self, u, V_omega,target,num_class):
        batch_size = u.size(0)
        V_omega = Variable(torch.from_numpy(V_omega).type(torch.FloatTensor).cuda())
        similarity_loss= similarityloss(u, target,num_class)
        constraint_loss = self.beta * (V_omega - u) ** 2
        loss = (similarity_loss.sum() + constraint_loss.sum()) / (self.num_train * batch_size)
        return loss




