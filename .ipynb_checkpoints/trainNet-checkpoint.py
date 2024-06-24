import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal

import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import wandb
import os

###
import seaborn as sns
# import pickle5 as pickale
import pickle

from net_model import Net
from util import Helper

class trainNet(Helper):
    
    def __init__(self, 
                 full_N = 52500, 
                 d_in = 15, 
                 d_mid = 3, 
                 d_out = 20, 
                 w = [500], 
                 Ns = [10000],
                 test_N = 2500,
                 trial = 3, 
                 num_eig = 15,
                 L=12, 
                 transfer = False, 
                 loss = None, 
                 w8_dk = 0,
                 requires_iNout = True,
                 path2data = None,
                 prjt_name = None):
        
        super().__init__(os.path.abspath(os.getcwd())+'/config.json')
        
        self.full_N = full_N
        self.d_in   = d_in 
        self.d_mid  = d_mid
        self.d_out  = d_out

        self.w = w
        self.trial = trial
        self.num_eig = 15

        self.Ns = Ns
        self.test_N = test_N
        self.num_Ns = len(self.Ns)
#         self.deep = deep
        self.L = L
        self.w8_dk = w8_dk
        self.transfer = transfer
        self.loss = loss
#         self.increment = increment
        
#         if not increment:
#             self.num_Ns = 1
            
        self.path2data = path2data
        self.prjt_name = prjt_name
        if requires_iNout == True:
            self.widths = [d_in] + L*w + [d_out]   
        elif requires_iNout == 'acc': #accodian structure; w is already well defined 
            self.widths = [d_in] + w + [d_out]
        else:
            self.widths = L*w
        
    def wandb_setup(self, num_data, trial):
        os.system("wandb login")
        wandb.init(project=self.prjt_name, entity="kchoi1230")
        wandb.run.name = 'testing_run_50k_{N}_{t}'.format(N = num_data, t = trial)
        wandb.run.save()
        
    def train_setup(self):
        self.train_costs = np.zeros([self.num_Ns, self.trial])
        self.test_costs = np.zeros([self.num_Ns, self.trial])
        self.norms = np.zeros([self.num_Ns, self.trial])
#         self.eig_Ws = np.zeros([self.num_Ns, self.trial, len(self.widths)-1, self.num_eig])
#         self.eig_alphas = np.zeros([self.num_Ns, self.trial, len(self.widths), self.num_eig])
        self.eig_Ws = []
        self.eig_alphas = []
        
    def load_dataset(self):

        self.X = torch.Tensor(pd.read_pickle(self.X_path+self.path2data)).cuda()
        self.Y = torch.Tensor(pd.read_pickle(self.Y_path+self.path2data)).cuda()
        
    def train_net(self, iN, N, t):
        self.net = Net(self.widths, transfer = self.transfer, loss = self.loss).cuda()
        
        self.net.train(self.X[:N],self.Y[:N],self.X[N:N+self.test_N],
                       self.Y[N:N+self.test_N],lr=1.5 * 0.001, 
                       weight_decay = self.w8_dk, 
                       epochs = 1200, num_batches=5)
        
        self.net.train(self.X[:N],self.Y[:N],self.X[N:N+self.test_N],
                       self.Y[N:N+self.test_N],lr=0.4 * 0.001, 
                       weight_decay = self.w8_dk + 0.002, 
                       epochs = 1200, num_batches=5)
        
        train_cost, test_cost, norm = self.net.train(self.X[:N],self.Y[:N],self.X[N:N+self.test_N],
                                                     self.Y[N:N+self.test_N],lr=0.1 * 0.001
                                                    , weight_decay = self.w8_dk + 0.0005, epochs = 1200, num_batches=5)
        
        self.train_costs[iN,t] = train_cost
        self.test_costs[iN,t] = test_cost
        self.norms[iN,t] = norm
#         self.eig_Ws[iN,t,:,:] = self.net.eigs_W(self.num_eig)
#         self.eig_alphas[iN,t,:,:] = self.net.eigs_alpha(self.num_eig)
        self.eig_Ws.append(self.net.eigs_W(self.num_eig))
        self.eig_alphas.append(self.net.eigs_alpha(self.num_eig))
        
    def train_loop(self):
        for iN, N in enumerate(self.Ns):
            for t in range(self.trial):
                self.wandb_setup(N, t)
                self.train_net(iN, N, t)
                wandb.finish()
        
#         else:
#             for t in range(self.trial):
#                 self.wandb_setup(self.full_N, t)
#                 self.train_net(0, self.full_N, t)
#                 wandb.finish()
                
    def save_Ws(self, path2data):
        pd.to_pickle(self.eig_Ws, self.eig_path + '_Ws_{path2data}'.format(path2data=path2data))
        pd.to_pickle(self.eig_alphas, self.eig_path + '_alphas_{path2data}'.format(path2data=path2data))
        
    def train_all(self):
        self.load_dataset()
        self.train_setup()
        self.train_loop()
        path_split = self.path2data.split('.')
        
        if self.L > 1:
            self.save_Ws('.'.join(path_split[:-1]) + '{Ns}_deep'.format(Ns=self.Ns[0]) + path_split[-1])
        elif self.L == 1: 
            self.save_Ws('.'.join(path_split[:-1]) + '{Ns}_shallow'.format(Ns=self.Ns[0]) + path_split[-1])
#         if self.deep:
#             self.save_Ws('.'.join(path_split[:-1]) + '{Ns}_deep'.format(Ns=self.Ns[0]) + path_split[-1])
#         else:
#             self.save_Ws('.'.join(path_split[:-1]) + '{Ns}_shallow'.format(Ns=self.Ns[0]) + path_split[-1])

    
if __name__ == "__main__":
    fire.Fire(trainNet)
    

    
    
    
    
    