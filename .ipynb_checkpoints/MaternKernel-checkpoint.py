import torch
# import gpytorch
import numpy as np
from scipy.special import kv, gamma
import logging
from tqdm import tqdm

class matern_Kernel():
    """Matern Kernel implementation based on scikit-learn"""
    def __init__(self, X, nu):
        self.X = X
        self.X_dim = X.shape[0]
        self.nu = nu
        
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%I:%M:%S %p', level=logging.INFO)
        self.logger = logging.getLogger('Kernel_logger')
        
    def compute(self):
        dists = np.zeros((self.X_dim, self.X_dim))
        
        for i in tqdm(range(self.X_dim)):
            dists[i,:] = np.sqrt(((self.X - self.X[i,:])**2).mean(-1))

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * np.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * np.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists**2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = np.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp**self.nu
            K *= kv(self.nu, tmp)
            
        np.fill_diagonal(K, 1)
        
        return K
    
    def sample(self, shape, Tikhonov = False):
        self.logger.info('Sampling Matern Kernel with nu: {nu}'.format(nu = self.nu))
        Generator = np.random.default_rng()
        K = self.compute()
        if Tikhonov:
            K[np.diag_indices_from(K)] += np.finfo(np.float32).eps
        Y = Generator.multivariate_normal(np.zeros(self.X_dim), K, size = (shape,), check_valid='warn', method='cholesky').T
        return Y
        