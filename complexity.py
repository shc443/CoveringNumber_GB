import json
import random
import pickle
import os
import glob
from scipy.optimize import curve_fit

import pandas as pd
import numpy as np

import itertools
from trainNet import trainNet
from net_model import Net

import torch
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as AGF
from torch.utils.data import Dataset
import torch.optim as optim

import matplotlib.pyplot as plt

from math import prod
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import os 
import wandb

class Net2(nn.Module):
    def __init__(self, dims, widths, nonlin=F.relu, skip=0):
        super(Net2, self).__init__()
        self.widths = widths
        self.dims = dims
        self.L = len(widths)
        self.nonlin=nonlin
        if skip==0:
            self.skip = [0 for i in widths]
        else:
            self.skip = skip
        self.linears = nn.ModuleList([nn.Linear(w1, w2) for w1,w2 in zip(widths[:-1],widths[1:])])
        self.alpha = [0 for _ in self.widths]
        self.pre_alpha = [0 for _ in self.widths]

    def shallow(self, z, layer):
        return self.out_linears[layer](self.nonlin(self.in_linears[layer](z)))
    
    def forward(self, x):
        self.alpha[0] = x
        for i, lin in enumerate(self.linears):
            self.pre_alpha[i+1] = lin(self.alpha[i])
            self.alpha[i+1] = self.nonlin(self.pre_alpha[i+1])
        return self.pre_alpha[len(self.widths)-1]

    def norms(self):
        return [0.5*torch.norm(self.linears[0].weight, p = 'fro')**2]+\
                [0.5*torch.norm(self.linears[idx].weight, p = 'nuc')**2 + 0.5*torch.norm(self.linears[idx-1].weight, p = 'nuc')**2 for idx in range(len(self.linears))[2::2]] +\
                [0.5*torch.norm(self.linears[-1].weight, p = 'fro')**2]

    def Lip_OP(self):
         return torch.Tensor([(torch.linalg.matrix_norm(linear.weight, ord=2)).item() for linear in self.linears]).reshape(-1,2).prod(axis=1)
        
    def ranks(self,atol=1e6,rtol=1e6):
        return ([self.dims[0]] 
                + [max(torch.linalg.matrix_rank(self.linears[idx].weight, atol=atol, rtol=rtol),
                      torch.linalg.matrix_rank(self.linears[idx-1].weight, atol=atol, rtol=rtol)).item() for idx in range(len(net.linears))[2::2]]
                + [self.dims[-1]])
    
    def stable_ranks(self):
        return ([self.dims[0]]
                + [(torch.norm(linear.weight)**2 
                 / torch.linalg.matrix_norm(linear.weight, ord=2)**2).item()  
                 for linear in net.linears[2::2]]
                + [self.dims[-1]])
    
    def complexity1(self, norms, lips, dims):
        return prod(lips) * sum([ n / l * (dm+dp)**0.5 for n,l,dm,dp in zip(norms, lips, dims[:-1], dims[1:])])
 
def nonlin(x):
    return F.relu(x) #* 0.7 + 0.3 * x

def myExpFunc(x, a, b):
    return a * np.power(x, b)