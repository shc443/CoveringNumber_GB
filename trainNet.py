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
    '''Tranier Module for numerical simulations'''

    def __init__(self,
                 full_N=52500,
                 d_in=15,
                 d_mid=3,
                 d_out=20,
                 w=[500],
                 Ns=[10000],
                 test_N=2500,
                 trial=3,
                 num_eig=15,
                 L=12,
                 transfer=False,
                 loss=None,
                 test_loss=None,
                 prof=False,
                 lr_scale = 1,
                 w8_dk=0,
                 requires_iNout=True,
                 path2data=None,
                 prjt_name=None,
                 wandb=False):
        '''
        :param full_N (int): total number of samples; in our paper, full_N is fixed as 52500 (50000 Training + 2500 Test)
        :param d_in (int): input dimension of dataset; 15
        :param d_mid (int): dimension of middle layers; 500 for DNN and AccNets or 50000 for Shallow net
        :param d_out (int): dimension of output layers; 20
        :param w (list of ints): list of widths dimensions; [15, 500, 500, ... , 500, 20] for DNN
        :param Ns (list of ints): list of input dataset size; sudo-logarithmic increment from 100 to full_n
        :param test_N (int): number of test samples; 2500
        :param trial (int): number of trials; 3
        :param num_eig (int): number of eigenvalues/eigenfunctions to find; 15
        :param L (int): depth of networks; 12
        :param transfer (bool, optional): whether to use transfer learning methods; False for this paper
        :param loss (str): loss function used during training; "L1" or "L2", all the others are considered as nuclear norm
        :param test_loss (str): loss function used during testing; "L1" or "L2", all the others are considered as nuclear norm
        :param prof (bool): whether to use Prof. Jacot's L1 code
        :param lr_scale (float): learning rate scale; 1; manage lr to be smaller or bigger for some cases
        :param w8_dk (float): weight decay rate
        :param requires_iNout (bool, str, optional): depreciated part; set to be 'acc'
        :param path2data (str): directory of dataset
        :param prjt_name (str, optional): name of project for WanDB;
        :param wandb (bool): whether to use WanDB for monintoring model performance; please set this as `False`
        '''
        super().__init__(os.path.abspath(os.getcwd()) + '/config.json')

        self.full_N = full_N
        self.d_in = d_in
        self.d_mid = d_mid
        self.d_out = d_out

        self.w = w
        self.trial = trial
        self.num_eig = 15

        self.Ns = Ns
        self.test_N = test_N
        self.num_Ns = len(self.Ns)
        #         self.deep = deep
        self.L = L
        self.lr_scale = lr_scale
        self.w8_dk = w8_dk
        self.transfer = transfer
        self.loss = loss
        self.test_loss = test_loss
        self.prof = prof

        self.path2data = path2data
        self.prjt_name = prjt_name
        self.wandb = wandb

        if requires_iNout == True:
            self.widths = [d_in] + L * w + [d_out]
        elif requires_iNout == 'acc':  # accodian structure; w is already well defined
            self.widths = [d_in] + w + [d_out]
        else:
            self.widths = L * w

    def wandb_setup(self, num_data, trial):
        os.system("wandb login")
        wandb.init(project=self.prjt_name, entity="kchoi1230")
        wandb.run.name = 'testing_run_50k_{N}_{t}'.format(N=num_data, t=trial)
        wandb.run.save()

    def train_setup(self):
        self.train_costs = np.zeros([self.num_Ns, self.trial])
        self.test_costs = np.zeros([self.num_Ns, self.trial])
        self.norms = np.zeros([self.num_Ns, self.trial])
        #         self.eig_Ws = np.zeros([self.num_Ns, self.trial, len(self.widths)-1, self.num_eig])
        #         self.eig_alphas = np.zeros([self.num_Ns, self.trial, len(self.widths), self.num_eig])
        self.eig_Ws = []
        self.eig_alphas = []
        self.complexity = []

    def load_dataset(self):

        self.X = torch.Tensor(pd.read_pickle(self.X_path + self.path2data)).cuda()
        self.Y = torch.Tensor(pd.read_pickle(self.Y_path + self.path2data)).cuda()

    def train_net(self, iN, N, t):
        self.net = Net(self.widths, transfer=self.transfer, loss=self.loss, test_loss=self.test_loss, prof=self.prof,
                       wandb=self.wandb).cuda()

        self.net.train(self.X[:N], self.Y[:N], self.X[N:N + self.test_N], self.Y[N:N + self.test_N], lr=1.5 * 0.001,
                       weight_decay=self.w8_dk, epochs=1200, num_batches=5)
        self.net.train(self.X[:N], self.Y[:N], self.X[N:N + self.test_N], self.Y[N:N + self.test_N], lr=0.4 * 0.001 * self.lr_scale,
                       weight_decay=self.w8_dk + 0.002, epochs=1200, num_batches=5)
        train_cost, test_cost, norm = self.net.train(self.X[:N], self.Y[:N], self.X[N:N + self.test_N],
                                                     self.Y[N:N + self.test_N], lr=0.1 * 0.001 * self.lr_scale
                                                     , weight_decay=self.w8_dk + 0.0005, epochs=1200, num_batches=5)

        self.train_costs[iN, t] = train_cost
        self.test_costs[iN, t] = test_cost
        self.norms[iN, t] = norm
        #         self.eig_Ws[iN,t,:,:] = self.net.eigs_W(self.num_eig)
        #         self.eig_alphas[iN,t,:,:] = self.net.eigs_alpha(self.num_eig)
        # self.eig_Ws.append(self.net.eigs_W(self.num_eig))
        # self.eig_alphas.append(self.net.eigs_alpha(self.num_eig))
        self.complexity.append(self.net.complexities(self.X[:N]))

    def train_loop(self):
        for iN, N in enumerate(self.Ns):
            for t in range(self.trial):
                if self.wandb:
                    self.wandb_setup(N, t)
                    self.train_net(iN, N, t)
                    wandb.finish()
                else:
                    self.train_net(iN, N, t)

    def save_Ws(self, path2data):
        pd.to_pickle(self.eig_Ws, self.eig_path + '_Ws_{path2data}'.format(path2data=path2data))
        pd.to_pickle(self.eig_alphas, self.eig_path + '_alphas_{path2data}'.format(path2data=path2data))

    def train_all(self):
        self.load_dataset()
        self.train_setup()
        self.train_loop()
        path_split = self.path2data.split('.')


if __name__ == "__main__":
    fire.Fire(trainNet)






