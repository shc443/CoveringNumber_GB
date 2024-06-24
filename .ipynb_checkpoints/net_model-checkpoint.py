import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal

import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import wandb

###
import seaborn as sns
# import pickle5 as pickale
import pickle


class Net(nn.Module):
    def __init__(self, widths, nonlin=F.relu, transfer = False, loss = None):
        super(Net, self).__init__()
        self.widths=widths
        self.nonlin=nonlin
        self.linears = nn.ModuleList([nn.Linear(w1, w2) for w1,w2 in zip(widths[:-1],widths[1:])])
        self.alpha = [0 for _ in self.widths]
        self.pre_alpha = [0 for _ in self.widths]
        self.opt = None
        self.transfer = transfer
        self.loss = loss #only for non-transfer

    def forward(self, x):
        self.alpha[0] = x
        for i, lin in enumerate(self.linears):
            self.pre_alpha[i+1] = lin(self.alpha[i])
            self.alpha[i+1] = self.nonlin(self.pre_alpha[i+1])
        return self.pre_alpha[len(self.widths)-1]

    def train(self, X_train, Y_train, X_test, Y_test, lr=0.5, weight_decay = 0.0, epochs = 2000, num_batches=1):
        N = X_train.shape[0] // num_batches
        if self.opt == None:
            self.opt = optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)
        
        for g in self.opt.param_groups:
            g['lr'] = lr
            g['weight_decay'] = weight_decay
        
        for t in range(epochs):
            for ib in range(num_batches):

                self.opt.zero_grad()
                
                if self.transfer:
                    N_half = N//2
                    X1 = X_train[N * ib : N * ib + N_half]
                    X2 = X_train[N * ib + N_half : N * (ib + 1)]

                    Y1 = self(X1)
                    Y2 = self(X2)
                    
                    cost1 = ((Y1[:,:Y1.shape[1]//2]
                              - Y_train[N * ib : N * ib + N_half, :Y1.shape[1]//2])**2).mean()
                    ###Todo" typo change cost2 to second half 
                    cost2 = ((Y2[:,Y2.shape[1]//2:]
                              - Y_train[N * ib + N_half : N * (ib + 1), Y2.shape[1]//2:])**2).mean()
                    cost = cost1 + cost2
                    
                else:
                    YY_train = self(X_train[N * ib:N*(ib+1)])
                    if self.loss == None:
                        cost = ((YY_train -  Y_train[N * ib:N*(ib+1)])**2).mean()
                    elif self.loss == 'L1':
                        # cost = torch.linalg.norm(YY_train -  Y_train[N * ib:N*(ib+1)], dim=1, ord=1)
                        cost = torch.abs((YY_train -  Y_train[N * ib:N*(ib+1)])).mean()
                    else:
                        cost = (YY_train -  Y_train[N * ib:N*(ib+1)]).norm(p='nuc')
                    
                cost.backward()
                self.opt.step()

            if t % 50 == 0:
                if self.transfer:
                    wandb.log({"cost": cost.item(),"cost1": cost1.item(),"cost2": cost2.item(), 
                           "test_cost": ((self(X_test) -  Y_test)**2).mean().item(), 
                           "norm": self.norm() / self.depth()})
                else: 
                    wandb.log({"cost": cost.item(), 
                               "test_cost": ((self(X_test) -  Y_test)**2).mean().item(), 
                               "norm": self.norm() / self.depth()})
#                 print(cost.item(), ((self(X_test) -  Y_test)**2).mean().item(), self.norm() / self.depth())
        return cost.item(), ((self(X_test) -  Y_test)**2).mean().item(), self.norm() / self.depth()

#     def train_transfer(self, X_train, Y_train, X_test, Y_test, lr=0.5, weight_decay = 0.0, epochs = 2000, num_batches=1):
#         N = X_train.shape[0] // num_batches
#         N_half = N//2
        
#         if self.opt == None:
#             self.opt = optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)
        
#         for g in self.opt.param_groups:
#             g['lr'] = lr
#             g['weight_decay'] = weight_decay
        
#         for t in range(epochs):
#             for ib in range(num_batches):

#                 self.opt.zero_grad()
                
#                 X1 = X_train[N * ib : N * ib + N_half]
#                 X2 = X_train[N * ib + N_half : N * (ib + 1)]
                
#                 Y1 = self(X1)
#                 Y2 = self(X2)
                
# #                 cost = ((YY_train -  Y_train[N * ib:N*(ib+1)])**2).mean()
#                 cost1 = ((Y1[:,:Y1.shape[1]//2]-Y_train[:,:Y1.shape[1]//2])**2).mean()
#                 cost2 = ((Y2[:,Y2.shape[1]//2:]-Y_train[:,Y2.shape[1]//2:])**2).mean()
#                 cost = cost1 + cost2
#                 cost.backward()
#                 self.opt.step()

#             if t % 50 == 0:
#                 wandb.log({"cost": cost.item(), 
#                            "test_cost": ((self(X_test) -  Y_test)**2).mean().item(), 
#                            "norm": self.norm() / self.depth()})
# #                 print(cost.item(), ((self(X_test) -  Y_test)**2).mean().item(), self.norm() / self.depth())
#         return cost.item(), ((self(X_test) -  Y_test)**2).mean().item(), self.norm() / self.depth()
    
    def nonlin_impact(self):
        nonlin_impact = np.zeros([len(self.alpha)])
        for i in range(1,len(self.alpha)):
            nonlin_impact[i] = ((self.alpha[i] - self.pre_alpha[i])**2).sum().item() / (self.pre_alpha[i]**2).sum().item()
        return nonlin_impact

    def eigs_alpha(self, num_eig=10):
#         eigs = np.zeros([len(self.alpha),num_eig])
        eigs = []
        
        for i,alpha in enumerate(self.alpha):
            S = torch.svd(alpha - 0*alpha.mean(0))[1].detach().cpu().numpy()
            if i==len(self.alpha)-1:
                S = torch.svd(self.pre_alpha[i] - 0*self.pre_alpha[i].mean(0))[1].detach().cpu().numpy()
            eigs.append(S[:num_eig])
        return eigs
    
    def eigs_W(self, num_eig=10):
#         eigs = np.zeros([len(self.linears),num_eig])
        eigs = []
        
        for i,lin in enumerate(self.linears):
            S = torch.svd(lin.weight)[1].detach().cpu().numpy()
#             eigs[i, :] = S[:num_eig]
            eigs.append(S[:num_eig])
        return eigs
    
    def norm(self):
        return sum([(p**2).sum().item() for p in self.parameters()])
    
    def depth(self):
        return len(self.widths)-1
    