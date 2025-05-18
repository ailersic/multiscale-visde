import warnings

import torch
from torch import nn, Tensor
from torch.func import vmap, jacfwd  # type: ignore

from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable
from jaxtyping import Float, jaxtyped
from beartype import beartype

from .autoencoder import VarEncoder
from .kernel import Kernel
from .utils import check_nn_dims
# ruff: noqa: F821, F722

@dataclass(frozen=True)
class LatentVarConfig:
    dim_mu: int
    dim_z: int

@runtime_checkable
class LatentVar(Protocol):
    @jaxtyped(typechecker=beartype)
    def __call__(self,
                 mu: Float[Tensor, "n_batch dim_mu"],
                 t: Float[Tensor, "n_batch 1"]
    ) -> tuple[Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"]
    ]:
        ...
    
    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"],
               t: Float[Tensor, "n_batch 1"]
    ) -> Float[Tensor, "... dim_z"]:
        ...

@runtime_checkable
class AmortizedLatentVar(Protocol):
    @jaxtyped(typechecker=beartype)
    def __call__(self,
                 mu: Float[Tensor, "n_batch dim_mu"],
                 t: Float[Tensor, "n_batch 1"],
    ) -> tuple[Float[Tensor, "n_batch dim_z"],
                Float[Tensor, "n_batch dim_z"],
                Float[Tensor, "n_batch dim_z"],
                Float[Tensor, "n_batch dim_z"]
    ]:
        ...
    
    @jaxtyped(typechecker=beartype)
    def form_window(self,
                    mu: Float[Tensor, "n_batch dim_mu"],
                    t: Float[Tensor, "n_batch 1"],
                    x_win: Float[Tensor, "n_batch n_win *shape_x"]
    ) -> None:
        ...
    
    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"],
               t: Float[Tensor, "n_batch 1"]
    ) -> Float[Tensor, "... dim_z"]:
        ...

class LatentVarGP(nn.Module):
    """Latent Gaussian process, learned as function of time"""

    def __init__(self,
                config: LatentVarConfig,
                net_mean: nn.Module,
                net_var: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.register_buffer("_empty_tensor", torch.empty(0))
        self.dim_mu = self.config.dim_mu
        self.dim_z = self.config.dim_z

        self.net_mean = net_mean
        self.net_var = net_var

        # check mean network dims
        check_nn_dims(self.net_mean,
                      ((self.dim_mu,), (1,)),
                      ((self.dim_z,),),
                      "Latent GP mean")

        # check var network dims
        check_nn_dims(self.net_var,
                      ((self.dim_mu,), (1,)),
                      ((self.dim_z,),),
                      "Latent GP variance")
    
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                t: Float[Tensor, "n_batch 1"]
    ) -> tuple[Float[Tensor, "n_batch dim_z"],
                Float[Tensor, "n_batch dim_z"],
                Float[Tensor, "n_batch dim_z"],
                Float[Tensor, "n_batch dim_z"]
    ]:
        z_mean = self.net_mean(mu, t)
        z_var = self.net_var(mu, t)
        assert torch.all(z_var > 0), "SDEVar: Variance must be positive"

        z_dmean = vmap(jacfwd(self.net_mean, argnums=1))(mu, t).squeeze(-1)
        z_dvar = vmap(jacfwd(self.net_var, argnums=1))(mu, t).squeeze(-1)

        return z_mean, z_var, z_dmean, z_dvar

    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"],
               t: Float[Tensor, "n_batch 1"]
    ) -> Float[Tensor, "... dim_z"]:
        z_mean, z_var, _, _ = self.forward(mu, t)
        z_stdev = torch.sqrt(z_var)

        eps_samples = torch.randn(t.shape[0], n_samples, self.dim_z, device=t.device)
        z_samples = (z_mean.unsqueeze(-2) + torch.mul(z_stdev.unsqueeze(-2), eps_samples)).flatten(0, 1)

        return z_samples

class AmortizedLatentVarGP(nn.Module):
    """Latent Gaussian process"""
    
    def __init__(self,
                config: LatentVarConfig,
                kernel: Kernel,
                encoder: VarEncoder
    ):
        super().__init__()
        self.config = config
        self.register_buffer("_empty_tensor", torch.empty(0))
        self.dim_z = self.config.dim_z

        self.kernel = kernel
        self.encoder = encoder
    
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    @jaxtyped(typechecker=beartype)
    def form_window(self,
                    mu: Float[Tensor, "n_batch dim_mu"],
                    t: Float[Tensor, "n_batch 1"],
                    x_win: Float[Tensor, "n_batch n_win *shape_x"]
    ) -> None:
        z_win_mean, z_win_var = self.encoder(mu, x_win)
        z_win_logvar = torch.log(z_win_var)

        assert torch.all(torch.isfinite(z_win_mean)), "SDEVar: Mean must be finite"
        assert torch.all(torch.isfinite(z_win_logvar)), "SDEVar: Log-variance must be finite"
        
        n_batch = t.shape[0]
        t_node = t - t[0]
        K = self.kernel(t_node, t_node)
        eye = torch.eye(n_batch, device=self.device)

        try:
            kern_chol = torch.linalg.cholesky(K + self.kernel.var * eye)
            assert torch.all(torch.isfinite(kern_chol)), "SDEVar: Cholesky decomposition is not finite"

        except (torch.linalg.LinAlgError, AssertionError):
            small_eig = torch.linalg.eigvals(K + self.kernel.var * eye).real.min()
            warnings.warn(f"Warning: kernel matrix is not positive definite. Smallest eigenvalue is {small_eig.item()}.")
            eps = 1e-6

            while True:
                try:
                    kern_chol = torch.linalg.cholesky(K + (self.kernel.var + eps + torch.abs(small_eig)) * eye)
                    assert torch.all(torch.isfinite(kern_chol)), "SDEVar: Cholesky decomposition is not finite"
                    warnings.warn(f"Warning: kernel matrix made positive definite with epsilon {eps}.")
                except (torch.linalg.LinAlgError, AssertionError):
                    eps *= 2
                else:
                    break
        
        frozen_kern = partial(self.kernel, t_node)

        def mean_interp(t_):
            return (frozen_kern(t_ - t[0]).T @ torch.cholesky_solve(z_win_mean, kern_chol)).squeeze(0)

        def logvar_interp(t_):
            return (frozen_kern(t_ - t[0]).T @ torch.cholesky_solve(z_win_logvar, kern_chol)).squeeze(0)

        self.z_mean_interp = mean_interp
        self.z_logvar_interp = logvar_interp
        self.zdot_mean_interp = vmap(jacfwd(mean_interp)) # vmap needed here to batch jacobian calculation
        self.zdot_logvar_interp = vmap(jacfwd(logvar_interp))

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"], # not used for amortized implementation
                t: Float[Tensor, "n_batch 1"],
    ) -> tuple[Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"]
    ]:
        z_mean = self.z_mean_interp(t)
        z_logvar = self.z_logvar_interp(t)

        zdot_mean = self.zdot_mean_interp(t).squeeze(-1)
        #zdot_mean = torch.diagonal(torch.autograd.functional.jacobian(self.z_mean_interp, t), dim1=0, dim2=2).squeeze(1).T

        zdot_logvar = self.zdot_logvar_interp(t).squeeze(-1)
        #zdot_logvar = torch.diagonal(torch.autograd.functional.jacobian(self.z_logvar_interp, t), dim1=0, dim2=2).squeeze(1).T

        return z_mean, z_logvar, zdot_mean, zdot_logvar

    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"], # not used for amortized implementation
               t: Float[Tensor, "n_batch 1"]
    ) -> Float[Tensor, "... dim_z"]:
        z_mean, z_logvar, _, _ = self.forward(mu, t)
        z_stdev = z_logvar.exp().sqrt()

        eps_samples = torch.randn(t.shape[0], n_samples, self.dim_z, device=t.device)
        z_samples = (z_mean.unsqueeze(-2) + torch.mul(z_stdev.unsqueeze(-2), eps_samples)).flatten(0, 1)

        return z_samples

class ParamFreeLatentVarGP(nn.Module):
    """Latent Gaussian process without additional trainable parameters"""
    
    def __init__(self,
                config: LatentVarConfig,
                encoder: VarEncoder
    ):
        super().__init__()
        self.config = config
        self.register_buffer("_empty_tensor", torch.empty(0))
        self.dim_z = self.config.dim_z

        self.encoder = encoder

        self.mu = None
        self.t = None

        self.z_mean = None
        self.z_logvar = None

        self.method = "fd"
    
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    def _savitzky_golay_coeffs(self,
                               t: Float[Tensor, "n_window_sg"],
                               z: Float[Tensor, "n_window_sg dim_z"],
                               order: int
    ) -> Tensor:
        """Compute Savitzky-Golay coefficients"""
        n_coeffs = order + 1
        n_window_sg = t.shape[0]
        dim_z = z.shape[1]

        assert n_window_sg >= n_coeffs, "SDEVar: Window size must be greater than order"

        A = torch.zeros(n_window_sg, n_coeffs, device=self.device)
        for i in range(n_window_sg):
            for j in range(n_coeffs):
                A[i, j] = t[i] ** j

        coeffs = torch.zeros(dim_z, n_coeffs, device=self.device)

        for i in range(dim_z):
            coeffs[i] = torch.linalg.lstsq(A, z[:, i]).solution

        return coeffs
    
    def _tvr_diff(self,
                 t: Float[Tensor, "n_window_sg"],
                 z: Float[Tensor, "n_window_sg dim_z"],
                 lr: float = 1e-2,
                 lambda_tv: float = 1e-3,
                 num_iters: int = 100
    ) -> Tensor:
        n_t, dim_z = z.shape
        dt = t[1:] - t[:-1]  # Time differences of shape [n_t-1, 1]

        # Compute finite difference approximation
        finite_diff = (z[1:] - z[:-1]) / dt

        # Initialize the derivative estimate as a trainable parameter
        dz_dt = torch.nn.Parameter(finite_diff.clone())

        optimizer = torch.optim.Adam([dz_dt], lr=lr)

        for _ in range(num_iters):
            optimizer.zero_grad()

            # TV regularization term: sum of absolute differences
            tv_term = torch.sum(torch.abs(dz_dt[1:] - dz_dt[:-1]))

            # Data fidelity term: squared error from finite difference
            loss = torch.sum((dz_dt - finite_diff) ** 2) + lambda_tv * tv_term

            loss.backward()
            optimizer.step()

        return dz_dt

    @jaxtyped(typechecker=beartype)
    def form_window(self,
                    mu: Float[Tensor, "n_batch dim_mu"],
                    t: Float[Tensor, "n_batch 1"],
                    x_win: Float[Tensor, "n_batch n_win *shape_x"]
    ) -> None:
        self.mu = mu
        self.t = t

        self.z_mean, z_var = self.encoder(mu, x_win)
        self.z_logvar = torch.log(z_var)

        assert torch.all(torch.isfinite(self.z_mean)), "SDEVar: Mean must be finite"
        assert torch.all(torch.isfinite(self.z_logvar)), "SDEVar: Log-variance must be finite"

        if self.method == "sg":
            # estimate zdot_mean and zdot_logvar by savitzky-golay filtering
            '''
            n_window_sg = 7
            order_sg = 3

            t_stamps = torch.stack([t[i:(len(t) + i - n_window_sg), 0] for i in range(n_window_sg)], dim=1)
            z_mean_stamps = torch.stack([self.z_mean[i:(len(t) + i - n_window_sg)] for i in range(n_window_sg)], dim=1)
            z_logvar_stamps = torch.stack([self.z_logvar[i:(len(t) + i - n_window_sg)] for i in range(n_window_sg)], dim=1)

            zdot_mean_coeffs = self.savitzky_golay_coeffs_vmap(t_stamps, z_mean_stamps, order_sg)
            zdot_mean_coeffs = torch.stack([zdot_mean_coeffs[:, 0] for i in range(n_window_sg//2)] +
                                        [zdot_mean_coeffs] +
                                        [zdot_mean_coeffs[:, -1] for i in range(n_window_sg//2)], dim=1)
            '''
            pass

        elif self.method == "tvr":
            # estimate zdot_mean and zdot_logvar by total variation regularization
            '''
            self.zdot_mean = self._tvr_diff(t, self.z_mean)
            self.zdot_logvar = self._tvr_diff(t, self.z_logvar)
            '''
            pass

        elif self.method == "fd":
            # estimate zdot_mean and zdot_logvar by finite difference
            self.zdot_mean = torch.zeros_like(self.z_mean)
            self.zdot_mean[:-1] = torch.diff(self.z_mean, dim=0) / torch.diff(self.t, dim=0)
            self.zdot_mean[-1] = self.zdot_mean[-2]

            self.zdot_logvar = torch.zeros_like(self.z_logvar)
            self.zdot_logvar[:-1] = torch.diff(self.z_logvar, dim=0) / torch.diff(self.t, dim=0)
            self.zdot_logvar[-1] = self.zdot_logvar[-2]

        else:
            raise ValueError("SDEVar: Invalid method for estimating latent variable derivatives")

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"], # not used for amortized implementation
                t: Float[Tensor, "n_batch 1"]
    ) -> tuple[Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"]
    ]:
        assert (self.z_mean is not None) and (self.z_logvar is not None), "SDEVar: Must form window before forward pass"
        assert torch.all(self.t == t) and torch.all(self.mu == mu), "SDEVar: Cannot use n_tquad != 0 for param-free latent var"

        return self.z_mean, self.z_logvar, self.zdot_mean, self.zdot_logvar

    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"], # not used for amortized implementation
               t: Float[Tensor, "n_batch 1"]
    ) -> Float[Tensor, "... dim_z"]:
        z_stdev = self.z_logvar.exp().sqrt()

        eps_samples = torch.randn(t.shape[0], n_samples, self.dim_z, device=t.device)
        z_samples = (self.z_mean.unsqueeze(-2) + torch.mul(z_stdev.unsqueeze(-2), eps_samples)).flatten(0, 1)

        return z_samples