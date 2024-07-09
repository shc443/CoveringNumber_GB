import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

import seaborn as sns
from math import prod
import pickle


class Net(nn.Module):
    """(Deep | AccNets | Shallow) Neural Network Model"""
    def __init__(self, widths, nonlin=F.relu, transfer=False, loss='L1', test_loss=None, prof=True, reduction='mean',
                 wandb=False):
        """
           :param widths(list of ints): widths of each layer in the network; e.g. [d_in, w, d_out] ~ [15, 500, 20] for a ShallowNet
           :param nonlin: nonlinear activation function; F.relu, F.leaky_relu, F.sigmoid, F.tanh
           :param transfer (bool, optional): whether to use transfer learning method:
           :param loss (str): type of loss function used during training; 'L1' or 'L2', all the others will be considered as nuclear norm
           :param test_loss (str): type of loss function used for validation (test); 'L1' or 'L2', all the others will be considered as nuclear norm
           :param prof (bool): whether to use Prof. Jacot's L1 definition
           :param reduction (str, optional): specifies the reduction to apply to the output; 'mean' or 'sum'
           :param wandb (bool, optional): whether to use wandb
        """
        super(Net, self).__init__()
        self.widths = widths
        self.nonlin = nonlin
        self.linears = nn.ModuleList([nn.Linear(w1, w2) for w1, w2 in zip(widths[:-1], widths[1:])])
        self.alpha = [0 for _ in self.widths]
        self.pre_alpha = [0 for _ in self.widths]
        self.opt = None
        self.transfer = transfer
        self.loss = loss  # only for non-transfer
        self.prof = prof
        self.test_loss = test_loss
        self.reduction = reduction
        self.wandb = wandb

    def forward(self, x):
        self.alpha[0] = x
        for i, lin in enumerate(self.linears):
            self.pre_alpha[i + 1] = lin(self.alpha[i])
            self.alpha[i + 1] = self.nonlin(self.pre_alpha[i + 1])
        return self.pre_alpha[len(self.widths) - 1]

    def cost(self, Y1, Y2):
        if self.loss == 'L2':
            cost = nn.MSELoss(reduction=self.reduction)(Y1, Y2)
        elif (self.loss == 'L1') and (not self.prof):
            cost = nn.L1Loss(reduction=self.reduction)(Y1, Y2)
        elif (self.loss == 'L1') and (self.prof):
            cost = (torch.sum((Y1 - Y2) ** 2, axis=0) ** 0.5).mean()
        elif self.loss == 'fro':
            cost = torch.norm(Y1 - Y2, p='fro')
        elif self.loss == 'nuc':
            cost = torch.norm(Y1 - Y2, p='nuc')
        else:
            cost = (Y1 - Y2).norm(p='nuc')
        return cost

    def train(self, X_train, Y_train, X_test, Y_test, lr=0.5, weight_decay=0.0, epochs=2000, num_batches=1):
        N = X_train.shape[0] // num_batches
        if self.opt == None:
            self.opt = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        for g in self.opt.param_groups:
            g['lr'] = lr
            g['weight_decay'] = weight_decay

        for t in range(epochs):
            for ib in range(num_batches):
                self.opt.zero_grad()
                YY_train = self(X_train[N * ib:N * (ib + 1)])
                cost = self.cost(YY_train, Y_train[N * ib:N * (ib + 1)])

                cost.backward()
                self.opt.step()

            if self.wandb:
                if t % 50 == 0:
                    test_cost = self.cost(self(X_test), Y_test)
                    if self.transfer:
                        wandb.log({"cost": cost.item(), "cost1": cost1.item(), "cost2": cost2.item(),
                                   "test_cost": test_cost.item(),
                                   "norm": self.norm() / self.depth()})
                    else:
                        wandb.log({"cost": cost.item(),
                                   "test_cost": test_cost.item(),
                                   "norm": self.norm() / self.depth()})

        return cost.item(), self.cost(self(X_test), Y_test).item(), self.norm() / self.depth()
        # Update Loss function : L1 norm along datapoint sum((self(X_test) -  Y_test))**2, axis = row)**0.5.mean()

    def nonlin_impact(self):
        nonlin_impact = np.zeros([len(self.alpha)])
        for i in range(1, len(self.alpha)):
            nonlin_impact[i] = ((self.alpha[i] - self.pre_alpha[i]) ** 2).sum().item() / (
                        self.pre_alpha[i] ** 2).sum().item()
        return nonlin_impact

    def eigs_alpha(self, num_eig=10):
        eigs = []

        for i, alpha in enumerate(self.alpha):
            S = torch.svd(alpha - 0 * alpha.mean(0))[1].detach().cpu().numpy()
            if i == len(self.alpha) - 1:
                S = torch.svd(self.pre_alpha[i] - 0 * self.pre_alpha[i].mean(0))[1].detach().cpu().numpy()
            eigs.append(S[:num_eig])
        return eigs

    def eigs_W(self, num_eig=10):
        eigs = []

        for i, lin in enumerate(self.linears):
            S = torch.svd(lin.weight)[1].detach().cpu().numpy()
            eigs.append(S[:num_eig])
        return eigs

    def norm(self):
        return sum([(p ** 2).sum().item() for p in self.parameters()])

    def depth(self):
        return len(self.widths) - 1

    def norms4(self):
        return [0.5 * torch.norm(self.linears[0].weight, p='fro') ** 2] + \
            [0.5 * torch.norm(self.linears[idx].weight, p='nuc') ** 2 + 0.5 * torch.norm(self.linears[idx - 1].weight,
                                                                                         p='nuc') ** 2 for idx in
             range(len(self.linears))[2::2]] + \
            [0.5 * torch.norm(self.linears[-1].weight, p='fro') ** 2]

    def Lip_OP(self):
        return torch.Tensor(
            [(torch.linalg.matrix_norm(linear.weight, ord=2)).item() for linear in self.linears]).reshape(-1, 2).prod(
            axis=1)

    def ranks(self, atol=0.1, rtol=0.1):
        return ([self.widths[0]]
                + [max(torch.linalg.matrix_rank(self.linears[idx].weight, atol=atol, rtol=rtol),
                       torch.linalg.matrix_rank(self.linears[idx - 1].weight, atol=atol, rtol=rtol)).item() for idx in
                   range(len(self.linears))[2::2]]
                + [self.widths[-1]])

    def stable_ranks(self):
        return ([self.widths[0]]
                + [(torch.norm(linear.weight) ** 2
                    / torch.linalg.matrix_norm(linear.weight, ord=2) ** 2).item()
                   for linear in self.linears[2::2]]
                + [self.widths[-1]])

    def complexity1(self, norms, lips, dims):
        return prod(lips) * sum([n / l * (dm + dp) ** 0.5 for n, l, dm, dp in zip(norms, lips, dims[:-1], dims[1:])])

    def complexities(self, X_train):
        norms = self.norms4()
        Lip_OPs = self.Lip_OP()

        # Compute complexity; complexity2 is based on stable ranks
        complexities1 = self.complexity1(norms, Lip_OPs, self.ranks()).item()
        complexities2 = self.complexity1(norms, Lip_OPs, self.stable_ranks()).item()

        # Compute variable terms in front of complexities
        rho = 1  # for L1
        b = max(torch.linalg.norm(torch.Tensor(X_train), dim=1))  # size of input ball
        constants = rho * b
        return constants * complexities1 / X_train.shape[0] ** 0.5, constants * complexities2 / X_train.shape[0] ** 0.5