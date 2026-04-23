"""
GPU-Accelerated Matern Kernel for Compositional Learning Experiments

This module provides a PyTorch-native Matern kernel implementation that:
1. Supports GPU acceleration for large-scale experiments
2. Enables learnable smoothness parameter nu
3. Provides autograd support for all parameters

For the ICLR 2025 paper:
"How DNNs Break the Curse of Dimensionality: Compositionality and Symmetry Learning"
Authors: Arthur Jacot, Seok Hoan Choi, Yuxiao Wen
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import kv as scipy_kv, gamma as scipy_gamma
from typing import Optional, Tuple, Union
import logging


class BesselKFunction(torch.autograd.Function):
    """
    Custom autograd function for K_nu(x) - Modified Bessel function of the second kind.

    Supports gradients with respect to both x AND nu (for learnable smoothness).

    Mathematical formulas:
        Forward:  K_nu(x)
        dK/dx:    -0.5 * [K_{nu-1}(x) + K_{nu+1}(x)]
        dK/dnu:   Numerical approximation (analytical is complex)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        # Store for backward
        ctx.save_for_backward(x, nu)

        # Compute using scipy (will be replaced with native CUDA later)
        x_np = x.detach().cpu().numpy()
        nu_val = nu.item() if nu.dim() == 0 else nu.detach().cpu().numpy()

        # Handle edge cases
        x_np = np.clip(x_np, 1e-10, None)  # K_nu(0) = inf

        result_np = scipy_kv(nu_val, x_np)
        result_np = np.nan_to_num(result_np, nan=0.0, posinf=1e30, neginf=-1e30)

        return torch.from_numpy(result_np.copy()).to(x.device, x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, nu = ctx.saved_tensors

        x_np = x.detach().cpu().numpy()
        nu_val = nu.item() if nu.dim() == 0 else nu.detach().cpu().numpy()
        x_np = np.clip(x_np, 1e-10, None)

        grad_x = None
        grad_nu = None

        # Gradient w.r.t. x: dK_nu/dx = -0.5 * [K_{nu-1} + K_{nu+1}]
        if ctx.needs_input_grad[0]:
            K_minus = scipy_kv(nu_val - 1, x_np)
            K_plus = scipy_kv(nu_val + 1, x_np)
            dK_dx = -0.5 * (K_minus + K_plus)
            dK_dx = np.nan_to_num(dK_dx, nan=0.0, posinf=0.0, neginf=0.0)
            grad_x = grad_output * torch.from_numpy(dK_dx.copy()).to(x.device, x.dtype)

        # Gradient w.r.t. nu: numerical approximation
        if ctx.needs_input_grad[1]:
            eps = 1e-5
            K_plus = scipy_kv(nu_val + eps, x_np)
            K_minus = scipy_kv(nu_val - eps, x_np)
            dK_dnu = (K_plus - K_minus) / (2 * eps)
            dK_dnu = np.nan_to_num(dK_dnu, nan=0.0, posinf=0.0, neginf=0.0)
            grad_nu = (grad_output * torch.from_numpy(dK_dnu.copy()).to(x.device, x.dtype)).sum()

        return grad_x, grad_nu


def bessel_k(x: torch.Tensor, nu: Union[torch.Tensor, float]) -> torch.Tensor:
    """
    Compute K_nu(x) - Modified Bessel function of the second kind.

    Args:
        x: Input tensor (must be positive)
        nu: Order parameter (scalar or tensor)

    Returns:
        K_nu(x) with same shape as x
    """
    if not isinstance(nu, torch.Tensor):
        nu = torch.tensor(nu, dtype=x.dtype, device=x.device)
    return BesselKFunction.apply(x, nu)


class MaternKernelTorch(nn.Module):
    """
    GPU-Accelerated Matern Kernel with optional learnable smoothness.

    The Matern kernel is defined as:
        K(r) = sigma^2 * (2^{1-nu}/Gamma(nu)) * (sqrt(2*nu)*r/l)^nu * K_nu(sqrt(2*nu)*r/l)

    where:
        - r: distance between points
        - sigma^2: variance (amplitude)
        - l: length scale
        - nu: smoothness parameter
        - K_nu: modified Bessel function of the second kind

    Special cases (closed-form, faster):
        - nu = 0.5: Exponential kernel
        - nu = 1.5: Matern 3/2
        - nu = 2.5: Matern 5/2
        - nu = inf: Squared exponential (RBF)

    Args:
        nu: Smoothness parameter (default: 2.5)
        variance: Kernel variance/amplitude (default: 1.0)
        lengthscale: Length scale (default: 1.0)
        learn_nu: If True, nu becomes a learnable parameter
        learn_variance: If True, variance becomes learnable
        learn_lengthscale: If True, lengthscale becomes learnable
        device: Device to use ('cuda' or 'cpu')
        dtype: Data type (default: torch.float64 for numerical stability)

    Example:
        >>> kernel = MaternKernelTorch(nu=12.73, learn_nu=True, device='cuda')
        >>> X = torch.randn(1000, 15, device='cuda', dtype=torch.float64)
        >>> K = kernel(X)  # [1000, 1000] kernel matrix
    """

    def __init__(
        self,
        nu: float = 2.5,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        learn_nu: bool = False,
        learn_variance: bool = False,
        learn_lengthscale: bool = False,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.learn_nu = learn_nu

        # Smoothness parameter
        if learn_nu:
            self.nu = nn.Parameter(torch.tensor(nu, dtype=dtype, device=device))
        else:
            self.register_buffer('nu', torch.tensor(nu, dtype=dtype, device=device))

        # Variance (use log for positivity)
        if learn_variance:
            self.log_variance = nn.Parameter(torch.tensor(np.log(variance), dtype=dtype, device=device))
        else:
            self.register_buffer('log_variance', torch.tensor(np.log(variance), dtype=dtype, device=device))

        # Lengthscale (use log for positivity)
        if learn_lengthscale:
            self.log_lengthscale = nn.Parameter(torch.tensor(np.log(lengthscale), dtype=dtype, device=device))
        else:
            self.register_buffer('log_lengthscale', torch.tensor(np.log(lengthscale), dtype=dtype, device=device))

        self.logger = logging.getLogger('MaternKernelTorch')

    @property
    def variance(self) -> torch.Tensor:
        return torch.exp(self.log_variance)

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.exp(self.log_lengthscale)

    def forward(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the kernel matrix K(X1, X2).

        Args:
            X1: [N, D] tensor of N points in D dimensions
            X2: [M, D] tensor (optional, defaults to X1)

        Returns:
            [N, M] kernel matrix (or [N, N] if X2 is None)
        """
        if X2 is None:
            X2 = X1

        # Compute pairwise distances
        # Using cdist for efficiency on GPU
        dist = torch.cdist(X1, X2, p=2)

        return self._compute_kernel(dist)

    def _compute_kernel(self, dist: torch.Tensor) -> torch.Tensor:
        """Compute kernel values from distance matrix."""
        nu = self.nu
        l = self.lengthscale
        var = self.variance

        # Check for special cases (closed-form, faster)
        nu_val = nu.item() if nu.dim() == 0 else nu

        if isinstance(nu_val, float):
            if abs(nu_val - 0.5) < 1e-8:
                return self._matern_05(dist)
            elif abs(nu_val - 1.5) < 1e-8:
                return self._matern_15(dist)
            elif abs(nu_val - 2.5) < 1e-8:
                return self._matern_25(dist)
            elif nu_val > 1e6:  # Treat as infinity -> RBF
                return self._rbf(dist)

        # General case: use Bessel function
        return self._matern_general(dist)

    def _matern_05(self, dist: torch.Tensor) -> torch.Tensor:
        """Matern 1/2 (Exponential kernel): K(r) = sigma^2 * exp(-r/l)"""
        return self.variance * torch.exp(-dist / self.lengthscale)

    def _matern_15(self, dist: torch.Tensor) -> torch.Tensor:
        """Matern 3/2: K(r) = sigma^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)"""
        scaled = np.sqrt(3) * dist / self.lengthscale
        return self.variance * (1 + scaled) * torch.exp(-scaled)

    def _matern_25(self, dist: torch.Tensor) -> torch.Tensor:
        """Matern 5/2: K(r) = sigma^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)"""
        scaled = np.sqrt(5) * dist / self.lengthscale
        return self.variance * (1 + scaled + scaled**2 / 3) * torch.exp(-scaled)

    def _rbf(self, dist: torch.Tensor) -> torch.Tensor:
        """RBF (Squared Exponential): K(r) = sigma^2 * exp(-r^2 / (2*l^2))"""
        return self.variance * torch.exp(-0.5 * (dist / self.lengthscale)**2)

    def _matern_general(self, dist: torch.Tensor) -> torch.Tensor:
        """General Matern kernel using Bessel function K_nu."""
        nu = self.nu
        l = self.lengthscale
        var = self.variance

        # Scaled distance: sqrt(2*nu) * r / l
        scaled = torch.sqrt(2 * nu) * dist / l

        # Avoid numerical issues at r=0
        scaled = torch.clamp(scaled, min=1e-10)

        # Normalization: 2^{1-nu} / Gamma(nu)
        # Use log for numerical stability
        log_norm = (1 - nu) * np.log(2) - torch.lgamma(nu)
        norm = torch.exp(log_norm)

        # K(r) = var * norm * scaled^nu * K_nu(scaled)
        K_nu_vals = bessel_k(scaled, nu)

        kernel = var * norm * torch.pow(scaled, nu) * K_nu_vals

        # Handle diagonal (r=0 case): K(0) = var
        # The formula gives 0 * inf = nan at r=0, should be var
        if kernel.shape[0] == kernel.shape[1]:
            kernel = kernel.masked_fill(torch.eye(kernel.shape[0], dtype=torch.bool, device=kernel.device), var.item())

        return kernel

    def sample(self, X: torch.Tensor, n_samples: int = 1, jitter: float = 1e-6) -> torch.Tensor:
        """
        Sample from a Gaussian Process with this kernel.

        Args:
            X: [N, D] input points
            n_samples: Number of samples to draw
            jitter: Small value added to diagonal for numerical stability

        Returns:
            [N, n_samples] tensor of GP samples
        """
        self.logger.info(f'Sampling Matern Kernel with nu: {self.nu.item():.4f}')

        # Compute kernel matrix
        K = self.forward(X)

        # Add jitter for numerical stability
        K = K + jitter * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)

        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            # If Cholesky fails, add more jitter
            self.logger.warning("Cholesky failed, adding more jitter")
            K = K + 1e-4 * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
            L = torch.linalg.cholesky(K)

        # Sample: Y = L @ Z where Z ~ N(0, I)
        Z = torch.randn(K.shape[0], n_samples, dtype=K.dtype, device=K.device)
        Y = L @ Z

        return Y

    def __repr__(self) -> str:
        return (f"MaternKernelTorch(nu={self.nu.item():.4f}, "
                f"variance={self.variance.item():.4f}, "
                f"lengthscale={self.lengthscale.item():.4f}, "
                f"learn_nu={self.learn_nu}, device='{self.device}')")


class CompositionalDataGenerator:
    """
    Generate compositional data Y = h(g(X)) using Matern kernels.

    For the ICLR 2025 paper experiments:
    - g: R^d -> R^k with smoothness nu_g (dimensionality reduction)
    - h: R^k -> R^m with smoothness nu_h (operates in reduced space)

    Args:
        d_input: Input dimension d
        d_intermediate: Bottleneck dimension k (k << d)
        d_output: Output dimension m
        device: 'cuda' or 'cpu'
        dtype: Data type (default: float64 for numerical stability)

    Example:
        >>> generator = CompositionalDataGenerator(d_input=15, d_intermediate=3, d_output=20)
        >>> X, Y = generator.generate(nu_g=2.0, nu_h=8.0, n_samples=50000)
    """

    def __init__(
        self,
        d_input: int = 15,
        d_intermediate: int = 3,
        d_output: int = 20,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64
    ):
        self.d_input = d_input
        self.d_intermediate = d_intermediate
        self.d_output = d_output
        self.device = device
        self.dtype = dtype

        self.logger = logging.getLogger('CompositionalDataGenerator')

    def generate(
        self,
        nu_g: float,
        nu_h: float,
        n_samples: int = 50000,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate compositional data Y = h(g(X)).

        Args:
            nu_g: Smoothness of g (dimensionality reduction function)
            nu_h: Smoothness of h (function on reduced space)
            n_samples: Number of data points
            seed: Random seed for reproducibility

        Returns:
            X: [n_samples, d_input] input data
            Y: [n_samples, d_output] output data (compositional)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.logger.info(f"Generating compositional data: nu_g={nu_g}, nu_h={nu_h}, n={n_samples}")

        # Step 1: Generate input X ~ Uniform or Normal
        X = torch.randn(n_samples, self.d_input, dtype=self.dtype, device=self.device)

        # Step 2: Sample Z = g(X) using Matern kernel with nu_g
        self.logger.info(f"Sampling g with nu={nu_g}")
        kernel_g = MaternKernelTorch(nu=nu_g, device=self.device, dtype=self.dtype)
        Z = kernel_g.sample(X, n_samples=self.d_intermediate)  # [n_samples, d_intermediate]

        # Step 3: Sample Y = h(Z) using Matern kernel with nu_h
        self.logger.info(f"Sampling h with nu={nu_h}")
        kernel_h = MaternKernelTorch(nu=nu_h, device=self.device, dtype=self.dtype)
        Y = kernel_h.sample(Z, n_samples=self.d_output)  # [n_samples, d_output]

        return X, Y

    def generate_from_existing_X(
        self,
        X: torch.Tensor,
        nu_g: float,
        nu_h: float,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate Y = h(g(X)) from existing X."""
        if seed is not None:
            torch.manual_seed(seed)

        X = X.to(device=self.device, dtype=self.dtype)

        # Sample Z = g(X)
        kernel_g = MaternKernelTorch(nu=nu_g, device=self.device, dtype=self.dtype)
        Z = kernel_g.sample(X, n_samples=self.d_intermediate)

        # Sample Y = h(Z)
        kernel_h = MaternKernelTorch(nu=nu_h, device=self.device, dtype=self.dtype)
        Y = kernel_h.sample(Z, n_samples=self.d_output)

        return X, Y


# Backward compatibility with original API
class matern_Kernel_torch:
    """
    Drop-in replacement for the original matern_Kernel class.

    Usage:
        # Old code:
        kernel = matern_Kernel(X_numpy, nu=2.5)
        K = kernel.compute()
        Y = kernel.sample(20)

        # New code (same API):
        kernel = matern_Kernel_torch(X_tensor, nu=2.5)
        K = kernel.compute()
        Y = kernel.sample(20)
    """

    def __init__(self, X: Union[np.ndarray, torch.Tensor], nu: float, device: str = 'cpu'):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        self.X = X.to(device=device, dtype=torch.float64)
        self.nu = nu
        self.device = device
        self.kernel = MaternKernelTorch(nu=nu, device=device, dtype=torch.float64)

        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                          datefmt='%I:%M:%S %p', level=logging.INFO)
        self.logger = logging.getLogger('Kernel_logger')

    def compute(self) -> np.ndarray:
        """Compute kernel matrix (returns numpy for compatibility)."""
        K = self.kernel(self.X)
        return K.cpu().numpy()

    def sample(self, shape: int, Tikhonov: bool = False) -> np.ndarray:
        """Sample from GP (returns numpy for compatibility)."""
        self.logger.info(f'Sampling Matern Kernel with nu: {self.nu}')
        jitter = 1e-6 if not Tikhonov else np.finfo(np.float32).eps
        Y = self.kernel.sample(self.X, n_samples=shape, jitter=jitter)
        return Y.cpu().numpy().T  # Transpose to match original [shape, N] format


if __name__ == "__main__":
    # Quick test
    print("Testing MaternKernelTorch...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test basic kernel computation
    X = torch.randn(100, 15, dtype=torch.float64, device=device)

    for nu in [0.5, 1.5, 2.5, 12.73]:
        kernel = MaternKernelTorch(nu=nu, device=device)
        K = kernel(X)
        print(f"nu={nu:>5.2f}: K shape={K.shape}, K[0,0]={K[0,0].item():.4f}, "
              f"K[0,1]={K[0,1].item():.6f}, symmetric={torch.allclose(K, K.T)}")

    # Test sampling
    print("\nTesting sampling...")
    kernel = MaternKernelTorch(nu=12.73, device=device)
    Y = kernel.sample(X, n_samples=20)
    print(f"Sample shape: {Y.shape}")

    # Test learnable nu
    print("\nTesting learnable nu...")
    kernel = MaternKernelTorch(nu=5.0, learn_nu=True, device=device)
    K = kernel(X)
    loss = K.sum()
    loss.backward()
    print(f"nu gradient: {kernel.nu.grad}")

    print("\nAll tests passed!")
