import pytorch_lightning as pl
import torch
import math

from dataclasses import dataclass
from torch import Tensor, nn
from torch.optim import lr_scheduler
from torchsde import sdeint
from jaxtyping import Float, jaxtyped
from beartype import beartype
from typing import Union

from .autoencoder import VarEncoder, VarDecoder
from .sdeprior import LatentDrift, LatentDispersion
from .likelihood import LogLike
from .sdevar import LatentVar, AmortizedLatentVar
from .utils import linterp
# ruff: noqa: F821, F722

#from torch.profiler import profile, record_function, ProfilerActivity

@dataclass(frozen=True)
class LatentSDEConfig:
    n_totaldata: int
    n_samples: int
    n_tquad: int
    n_warmup: int
    n_transition: int
    lr: float
    lr_sched_freq: int

class LatentSDE(pl.LightningModule):
    """Variational inference of latent SDE"""

    _empty_tensor: Tensor  # empty tensor to get device
    _iter_count: int  # iteration count
    config: LatentSDEConfig
    encoder: VarEncoder
    decoder: VarDecoder
    drift: LatentDrift
    dispersion: LatentDispersion
    loglikelihood: LogLike
    latentvar: Union[LatentVar, AmortizedLatentVar]

    def __init__(
        self,
        config: LatentSDEConfig,
        encoder: VarEncoder,
        decoder: VarDecoder,
        drift: LatentDrift,
        dispersion: LatentDispersion,
        loglikelihood: LogLike,
        latentvar: Union[LatentVar, AmortizedLatentVar],
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.drift = drift
        self.dispersion = dispersion
        self.loglikelihood = loglikelihood
        self.latentvar = latentvar

        self.register_buffer("_empty_tensor", torch.empty(0))
        self._iter_count = 0

    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    @jaxtyped(typechecker=beartype)
    def _log_likelihood(self,
                        mu: Float[Tensor, "n_batch dim_mu"],
                        t: Float[Tensor, "n_batch 1"],
                        x: Float[Tensor, "n_batch *shape_x"]
    ) -> Float[Tensor, ""]:
        """Compute expected log-likelihood"""

        n_batch = mu.shape[0]

        # sample from variational latent state distribution
        n_samples = self.config.n_samples
        z_samples = self.latentvar.sample(n_samples, mu, t)
        mu_samples = mu.repeat(n_samples, 1)

        assert torch.isfinite(z_samples).all(), f"Log likelihood: Samples from latent var distribution have nan or inf values: {z_samples}"

        # decode samples
        x_mean, x_var = self.decoder(mu_samples, z_samples)
        x_mean = x_mean.unflatten(0, (n_batch, n_samples))
        x_var = x_var.unflatten(0, (n_batch, n_samples))

        assert torch.isfinite(x_mean).all(), f"Log likelihood: Decoder mean has nan or inf values: {x_mean}"
        assert torch.isfinite(x_var).all(), f"Log likelihood: Decoder variance has nan or inf values: {x_var}"
        assert torch.nonzero(x_var <= 0).shape[0] == 0, f"Log likelihood: Decoder variance has non-positive values: {x_var[x_var <= 0]}"

        # compute expected log-likelihood
        expected_log_like = self.loglikelihood(x.flatten(1),
                                               x_mean.flatten(2),
                                               x_var.flatten(2)
        ).mean(1).sum(0)

        return expected_log_like

    @jaxtyped(typechecker=beartype)
    def _residual_norm(self,
                       mu_batch: Float[Tensor, "n_batch dim_mu"],
                       t_batch: Float[Tensor, "n_batch 1"],
                       f_batch: Float[Tensor, "n_batch dim_f"]
    ) -> Tensor:
        """Compute drift residual norm at arbitrary quadrature points"""

        n_samples = self.config.n_samples

        # if config n_tquad is zero, sample at data time stamps (this is mandatory for param-free latent var)
        # otherwise, interpolate mu and f at n_tquad time samples
        if self.config.n_tquad == 0:
            t, mu, f = t_batch, mu_batch, f_batch
        else:
            t = torch.linspace(t_batch[0, 0], t_batch[-1, 0], self.config.n_tquad, device=self.device).unsqueeze(-1)
            mu, f = linterp(mu_batch, t_batch.flatten(), t.flatten()), linterp(f_batch, t_batch.flatten(), t.flatten())

        n_tquad = t.shape[0]

        # sample from variational latent state distribution
        z_mean, z_logvar, z_dmean, z_dlogvar = self.latentvar(mu, t)
        dim_z = z_mean.shape[-1]

        # repeat n_samples times to match z_samples shape
        z_mean_samples = z_mean.repeat_interleave(n_samples, dim=0)
        z_var_samples = z_logvar.exp().repeat_interleave(n_samples, dim=0)
        z_dmean_samples = z_dmean.repeat_interleave(n_samples, dim=0)
        z_dlogvar_samples = z_dlogvar.repeat_interleave(n_samples, dim=0)

        assert torch.isfinite(z_mean_samples).all(), f"Residual norm: Mean of latent var distribution has nan or inf values: {z_mean_samples}"
        assert torch.isfinite(z_var_samples).all(), f"Residual norm: Variance of latent var distribution has nan or inf values: {z_var_samples}"
        assert torch.nonzero(z_var_samples <= 0).shape[0] == 0, f"Residual norm: Variance of latent var distribution has non-positive values: {z_var_samples[z_var_samples <= 0]}"

        # generate z samples
        z_samples = self.latentvar.sample(n_samples, mu, t)
        t_samples = t.repeat_interleave(n_samples, dim=0)
        f_samples = f.repeat_interleave(n_samples, dim=0)
        mu_samples = mu.repeat_interleave(n_samples, dim=0)

        assert torch.isfinite(z_samples).all(), f"Residual norm: Samples from latent var distribution have nan or inf values: {z_samples}"

        # compute brownian variance
        dis_samples = self.dispersion(mu_samples, t_samples)
        brownian_var_samples = dis_samples.pow(2)
        assert brownian_var_samples.shape == (n_tquad*n_samples, dim_z)

        assert torch.isfinite(brownian_var_samples).all(), f"Residual norm: Brownian variance has nan or inf values: {brownian_var_samples}"
        assert torch.nonzero(brownian_var_samples <= 0).shape[0] == 0, f"Residual norm: Brownian variance has non-positive values: {brownian_var_samples[brownian_var_samples <= 0]}"

        # compute residual
        B = 0.5 * (brownian_var_samples.div(z_var_samples) - z_dlogvar_samples)
        assert B.shape == (n_tquad*n_samples, dim_z)
        samples_diff = torch.mul(B, z_mean_samples - z_samples)
        derivative_diff = z_dmean_samples - self.drift(mu_samples, t_samples, z_samples, f_samples)
        res = samples_diff + derivative_diff

        # compute residual norm
        res_norm = 0.5 * res.pow(2).div(brownian_var_samples).sum(-1)
        T = t[-1, 0] - t[0, 0]

        return res_norm.mean(0) * T

    @jaxtyped(typechecker=beartype)
    def _compute_elbo_components(self,
                                mu:     Float[Tensor, "n_batch dim_mu"],
                                t:      Float[Tensor, "n_batch 1"],
                                x_win:  Float[Tensor, "n_batch n_win *shape_x"],
                                x:      Float[Tensor, "n_batch *shape_x"],
                                f:      Float[Tensor, "n_batch dim_f"]
    ) -> tuple[Tensor, Tensor, Tensor]:

        # resample parameters
        self.encoder.resample_params()
        self.decoder.resample_params()
        self.drift.resample_params()
        self.dispersion.resample_params()

        # form latent distribution over window
        if isinstance(self.latentvar, AmortizedLatentVar):
            self.latentvar.form_window(mu, t, x_win)

        # log-likelihood
        log_like = self._log_likelihood(mu, t, x)

        # drift residual
        drift_res = self._residual_norm(mu, t, f)

        # kl-divergence
        kl_div = (self.encoder.kl_divergence() +
                self.decoder.kl_divergence() +
                self.drift.kl_divergence() +
                self.dispersion.kl_divergence())
        
        assert torch.isfinite(log_like), f"Log likelihood is nan or inf: {log_like.item()}"
        assert torch.isfinite(drift_res), f"Drift residual is nan or inf: {drift_res.item()}"
        assert torch.isfinite(kl_div), f"KL divergence is nan or inf: {kl_div.item()}"

        # normalization, approximates ELBO / (n_totaldata * dim_x)
        n_batch = mu.shape[0]
        n_totaldata = self.config.n_totaldata
        dim_x = x.view(n_batch, -1).shape[-1]

        norm_log_like = log_like / (n_batch * dim_x)
        norm_drift_res = drift_res / (n_batch * dim_x)
        norm_kl_div = kl_div / (n_totaldata * dim_x)

        return norm_log_like, norm_drift_res, norm_kl_div

    @jaxtyped(typechecker=beartype)
    def training_step(self,
                      batch: list[Tensor],
                      batch_idx: int
    ) -> Tensor:
        """Calculate the loss for a single training step"""

        self._iter_count += 1
        if self.config.n_transition == 0:
            beta = 0.0 if self._iter_count < self.config.n_warmup else 1.0
        else:
            beta = max(min((self._iter_count - self.config.n_warmup) / self.config.n_transition, 1.0), 0.0)
        self.log("beta", beta)
        
        batch = [b.to(self.device) for b in batch]
        mu, t, x_win, x, f = batch

        # t.shape can either be (n_batch,) or (n_batch, 1); this ensures the latter
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        # compute losses
        log_like, drift_res, kl_div = self._compute_elbo_components(mu, t, x_win, x, f)

        self.log("train/log_like", log_like.item())
        self.log("train/drift_res", drift_res.item())
        self.log("train/kl_div", kl_div.item())

        warmup_elbo = log_like - beta * (drift_res + kl_div) # with beta scaling
        raw_elbo = log_like - drift_res - kl_div # true elbo

        self.log("train/loss", -warmup_elbo.item())
        self.log("train/raw_loss", -raw_elbo.item(), prog_bar=True)

        return -warmup_elbo
    
    def validation_step(self,
                        batch: list[Tensor],
                        batch_idx: int
    ) -> Tensor:
        """Calculate the loss for a single validation step"""

        batch = [b.to(self.device) for b in batch]
        mu, t, x_win, x, f = batch

        # t.shape can either be (n_batch,) or (n_batch, 1); this ensures the latter
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        # compute losses
        log_like, drift_res, kl_div = self._compute_elbo_components(mu, t, x_win, x, f)

        self.log("val/log_like", log_like.item())
        self.log("val/drift_res", drift_res.item())
        self.log("val/kl_div", kl_div.item())

        raw_elbo = log_like - drift_res - kl_div

        self.log("val/raw_loss", -raw_elbo.item(), prog_bar=True)

        # compute error over minibatch
        temp_sde = SDE(self.drift, self.dispersion, mu[0:1], t, f)
        z_enc_mean, _ = self.encoder(mu, x_win)
        z_int = sdeint(temp_sde, z_enc_mean[0:1], t, method="euler").squeeze(1)
        x_dec_mean, _ = self.decoder(mu, z_int)

        x_err = (x_dec_mean - x).pow(2).flatten(1).sum(1)
        x_norm = x.pow(2).flatten(1).sum(1)
        rmsre = (x_err/x_norm).mean().sqrt()

        self.log("val/rmsre", rmsre.item(), prog_bar=True)

        return -raw_elbo

    def configure_optimizers(self):
        '''Recommended default optimizer and scheduler for training'''
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        gamma = math.exp(math.log(0.9) / self.config.lr_sched_freq)
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "interval": "step",  # call scheduler after every train step
                "frequency": 1,
            },
        }

class SDE(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, drift, dispersion, test_mu, test_t, test_f):
        super().__init__()
        self.drift = drift
        self.dispersion = dispersion
        self.test_mu = test_mu
        self.test_t = test_t
        self.test_f = test_f
    
    def _forcing(self, t):
        return linterp(self.test_f.squeeze(0), self.test_t.squeeze(), t.squeeze(1))

    # Drift
    def f(self, t, y):
        t = t.repeat(y.shape[0], 1)
        mu = self.test_mu.repeat(y.shape[0], 1)
        fi = self._forcing(t)
        return self.drift(mu, t, y, fi)  # shape (n_batch, dim_z)

    # Diffusion
    def g(self, t, y):
        t = t.repeat(y.shape[0], 1)
        mu = self.test_mu.repeat(y.shape[0], 1)
        return self.dispersion(mu, t)  # shape (n_batch, dim_z)
