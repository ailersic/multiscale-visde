import torch
from torch import nn, Tensor
from typing import Protocol, runtime_checkable
from jaxtyping import Float, jaxtyped
from dataclasses import dataclass
from beartype import beartype

from .utils import check_nn_dims
# ruff: noqa: F821, F722

@dataclass(frozen=True)
class LatentDriftConfig:
    dim_mu: int
    dim_z: int
    dim_f: int

@dataclass(frozen=True)
class LatentDispersionConfig:
    dim_mu: int
    dim_z: int

@runtime_checkable
class LatentDrift(Protocol):
    def resample_params(self) -> None:
        ...
    
    def kl_divergence(self) -> Tensor:
        ...
    
    @jaxtyped(typechecker=beartype)
    def __call__(self,
                 mu: Float[Tensor, "n_batch dim_mu"],
                 t: Float[Tensor, "n_batch 1"],
                 z: Float[Tensor, "n_batch dim_z"],
                 f: Float[Tensor, "n_batch dim_f"]
    ) -> Float[Tensor, "n_batch dim_z"]:
        ...

@runtime_checkable
class LatentDispersion(Protocol):
    def resample_params(self) -> None:
        ...
    
    def kl_divergence(self) -> Tensor:
        ...
    
    @jaxtyped(typechecker=beartype)
    def __call__(self,
                 mu: Float[Tensor, "n_batch dim_mu"],
                 t: Float[Tensor, "n_batch 1"]
    ) -> Float[Tensor, "n_batch dim_z"]:
        ...

class LatentDriftNoPrior(nn.Module):
    """Latent drift function, parameterized as a neural network"""

    _empty_tensor: Tensor  # empty tensor to get device
    config: LatentDriftConfig
    net: nn.Module
    dim_z: int

    def __init__(self, config: LatentDriftConfig, driftnet: nn.Module):
        super().__init__()
        self.config = config
        self.register_buffer("_empty_tensor", torch.empty(0))

        self.net = driftnet
        self.dim_mu = self.config.dim_mu
        self.dim_z = self.config.dim_z
        self.dim_f = self.config.dim_f

        # check drift network dims
        check_nn_dims(self.net,
                      ((self.dim_mu,), (1,), (self.dim_z,), (self.dim_f,)),
                      ((self.dim_z,),),
                      "Latent drift")
    
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    def resample_params(self) -> None:
        pass
    
    def kl_divergence(self) -> Tensor:
        return torch.tensor(0.0)

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                t: Float[Tensor, "n_batch 1"],
                z: Float[Tensor, "n_batch dim_z"],
                f: Float[Tensor, "n_batch dim_f"]
    ) -> Float[Tensor, "n_batch dim_z"]:
        return self.net(mu, t, z, f)

class LatentDispersionNoPrior(nn.Module):
    """Latent dispersion function, parameterized as a neural network"""

    _empty_tensor: Tensor  # empty tensor to get device
    config: LatentDispersionConfig
    net: nn.Module
    dim_mu: int
    dim_z: int

    def __init__(self, config: LatentDispersionConfig, dispnet: nn.Module):
        super().__init__()
        self.config = config
        self.register_buffer("_empty_tensor", torch.empty(0))

        self.net = dispnet
        self.dim_mu = self.config.dim_mu
        self.dim_z = self.config.dim_z

        # check dispersion network dims
        check_nn_dims(self.net,
                      ((self.dim_mu,), (1,)),
                      ((self.dim_z,),),
                      "Latent dispersion")
        
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    def resample_params(self) -> None:
        pass
    
    def kl_divergence(self) -> Tensor:
        return torch.tensor(0.0)

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                t: Float[Tensor, "n_batch 1"]
    ) -> Float[Tensor, "n_batch dim_z"]:
        return self.net(mu, t)