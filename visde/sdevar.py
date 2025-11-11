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
        self.n_pad = 8  # number of padding points on each side of the window
        self.pad_window = False

    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    def pad_window(self,
                   x_win: Float[Tensor, "n_batch n_win *shape_x"],
                   n_pad: int
    ) -> Float[Tensor, "(n_batch + 2*n_pad) n_win *shape_x"]:
        if n_pad == 0:
            return x_win
        
        dx_0 = x_win[1:2] - x_win[0:1]
        incr_range_0 = torch.arange(n_pad, 0, -1, device=x_win.device).view(-1, *([1] * (x_win.ndim - 1)))
        left_pad = x_win[0:1] - incr_range_0 * dx_0

        dx_end = x_win[-1:] - x_win[-2:-1]
        incr_range_end = torch.arange(1, n_pad + 1, device=x_win.device).view(-1, *([1] * (x_win.ndim - 1)))
        right_pad = x_win[-1:] + incr_range_end * dx_end

        x_win_padded = torch.cat([left_pad, x_win, right_pad], dim=0)
        return x_win_padded

    def pad_time(self,
                 t: Float[Tensor, "n_batch 1"],
                 n_pad: int
    ) -> Float[Tensor, "(n_batch + 2*n_pad) 1"]:
        if n_pad == 0:
            return t
        dt = t[1] - t[0]
        left_pad = (t[0] - torch.arange(n_pad, 0, -1, device=t.device).unsqueeze(-1) * dt).to(t.dtype)
        right_pad = (t[-1] + torch.arange(1, n_pad + 1, device=t.device).unsqueeze(-1) * dt).to(t.dtype)
        t_padded = torch.cat([left_pad, t, right_pad], dim=0)
        return t_padded

    @jaxtyped(typechecker=beartype)
    def form_window(self,
                    mu: Float[Tensor, "n_batch dim_mu"],
                    t: Float[Tensor, "n_batch 1"],
                    x_win: Float[Tensor, "n_batch n_win *shape_x"]
    ) -> None:
        """Form interpolation window and precompute interpolation functions"""
        if self.pad_window:
            x_win = self.pad_window(x_win, self.n_pad)
            mu = self.pad_window(mu.unsqueeze(1), self.n_pad).squeeze(1)
            t = self.pad_time(t, self.n_pad)

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
            if t_.ndim == 1:
                t_ = t_.unsqueeze(-1)
            return (frozen_kern(t_ - t[0]).T @ torch.cholesky_solve(z_win_mean, kern_chol)).squeeze(0)

        def logvar_interp(t_):
            if t_.ndim == 1:
                t_ = t_.unsqueeze(-1)
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

        self.method = "sg"
        self.savitzky_golay_coeffs_vmap = torch.vmap(self._savitzky_golay_coeffs, in_dims=(1, 1), out_dims=(1,))
        self.sg_n_win_default = 25
        self.sg_order = 3
    
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    def _savitzky_golay_coeffs(self,
                               t: Float[Tensor, "sg_n_win"],
                               z: Float[Tensor, "sg_n_win dim_z"]
    ) -> Float[Tensor, "n_coeffs dim_z"]:
        """Compute Savitzky-Golay coefficients"""
        n_coeffs = self.sg_order + 1
        dim_z = z.shape[1]

        A = torch.pow(t.unsqueeze(1), torch.arange(n_coeffs).unsqueeze(0).to(z.device))
        coeffs = torch.linalg.lstsq(A.unsqueeze(0).expand(dim_z, -1, -1), z.T.unsqueeze(-1)).solution.squeeze(-1).T

        return coeffs

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

        if self.method == "sg":
            # estimate zdot_mean and zdot_logvar by Savitzky-Golay filtering
            n_batch = t.shape[0]
            sg_n_win = min(n_batch, self.sg_n_win_default)
            deriv_coeff_vec = torch.arange(1, self.sg_order + 1).reshape(-1, 1, 1).to(t.device)

            t_stamps = torch.stack([t[i:(i + sg_n_win), 0] for i in range(n_batch - sg_n_win + 1)], dim=1)
            z_mean_stamps = torch.stack([self.z_mean[i:(i + sg_n_win)] for i in range(n_batch - sg_n_win + 1)], dim=1)
            z_logvar_stamps = torch.stack([self.z_logvar[i:(i + sg_n_win)] for i in range(n_batch - sg_n_win + 1)], dim=1)

            z_mean_coeffs = self.savitzky_golay_coeffs_vmap(t_stamps, z_mean_stamps)
            z_logvar_coeffs = self.savitzky_golay_coeffs_vmap(t_stamps, z_logvar_stamps)

            if sg_n_win == n_batch:
                z_mean_coeffs = z_mean_coeffs.expand(-1, sg_n_win, -1)
                z_logvar_coeffs = z_logvar_coeffs.expand(-1, sg_n_win, -1)
            else:
                z_mean_coeffs = torch.cat([z_mean_coeffs[:, 0].unsqueeze(1) for i in range(sg_n_win//2)] +
                                        [z_mean_coeffs] +
                                        [z_mean_coeffs[:, -1].unsqueeze(1) for i in range(sg_n_win//2)], dim=1)
                z_logvar_coeffs = torch.cat([z_logvar_coeffs[:, 0].unsqueeze(1) for i in range(sg_n_win//2)] +
                                            [z_logvar_coeffs] +
                                            [z_logvar_coeffs[:, -1].unsqueeze(1) for i in range(sg_n_win//2)], dim=1)
            
            zdot_mean_coeffs = z_mean_coeffs[1:] * deriv_coeff_vec
            zdot_logvar_coeffs = z_logvar_coeffs[1:] * deriv_coeff_vec

            t_poly = torch.pow(t.reshape(1, -1, 1), torch.arange(self.sg_order + 1).to(t.device).reshape(-1, 1, 1))
            self.z_mean = torch.sum(t_poly * z_mean_coeffs, dim=0)
            self.zdot_mean = torch.sum(t_poly[:-1] * zdot_mean_coeffs, dim=0)
            self.z_logvar = torch.sum(t_poly * z_logvar_coeffs, dim=0)
            self.zdot_logvar = torch.sum(t_poly[:-1] * zdot_logvar_coeffs, dim=0)

        elif self.method == "fd":
            # estimate zdot_mean and zdot_logvar by finite difference
            self.zdot_mean = torch.zeros_like(self.z_mean)
            self.zdot_mean[1:-1] = (self.z_mean[2:] - self.z_mean[:-2]) / (self.t[2:] - self.t[:-2])
            self.zdot_mean[-1] = (self.z_mean[-1] - self.z_mean[-2]) / (self.t[-1] - self.t[-2])
            self.zdot_mean[0] = (self.z_mean[1] - self.z_mean[0]) / (self.t[1] - self.t[0])

            self.zdot_logvar = torch.zeros_like(self.z_logvar)
            self.zdot_logvar[1:-1] = (self.z_logvar[2:] - self.z_logvar[:-2]) / (self.t[2:] - self.t[:-2])
            self.zdot_logvar[-1] = (self.z_logvar[-1] - self.z_logvar[-2]) / (self.t[-1] - self.t[-2])
            self.zdot_logvar[0] = (self.z_logvar[1] - self.z_logvar[0]) / (self.t[1] - self.t[0])

        else:
            raise ValueError("SDEVar: Invalid method for estimating latent variable derivatives")

        assert torch.all(torch.isfinite(self.z_mean)), "SDEVar: Mean must be finite"
        assert torch.all(torch.isfinite(self.z_logvar)), "SDEVar: Log-variance must be finite"
        assert torch.all(torch.isfinite(self.zdot_mean)), "SDEVar: Mean derivative must be finite"
        assert torch.all(torch.isfinite(self.zdot_logvar)), "SDEVar: Log-variance derivative must be finite"

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