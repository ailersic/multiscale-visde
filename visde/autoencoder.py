import torch
from torch import Tensor, nn

from typing import Protocol, runtime_checkable
from jaxtyping import Float, jaxtyped
from beartype import beartype
from dataclasses import dataclass

from .utils import check_nn_dims
# ruff: noqa: F821, F722

@dataclass(frozen=True)
class VarAutoencoderConfig:
    dim_mu: int
    dim_x: int
    dim_z: int
    n_win: int
    shape_x: tuple[int, ...]

@runtime_checkable
class VarEncoder(Protocol):
    def resample_params(self) -> None:
        ...
    
    def kl_divergence(self) -> Tensor:
        ...

    @jaxtyped(typechecker=beartype)
    def __call__(self,
                 mu: Float[Tensor, "n_batch dim_mu"],
                 x_win: Float[Tensor, "n_batch n_win *shape_x"]
    ) -> tuple[Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"]
    ]:
        ...

    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"],
               x_win: Float[Tensor, "n_batch n_win *shape_x"]
    ) -> Float[Tensor, "... dim_z"]:
        ...

@runtime_checkable
class VarDecoder(Protocol):
    def resample_params(self) -> None:
        ...
    
    def kl_divergence(self) -> Tensor:
        ...
    
    @jaxtyped(typechecker=beartype)
    def __call__(self,
                 mu: Float[Tensor, "n_batch dim_mu"],
                 z: Float[Tensor, "n_batch dim_z"]
    ) -> tuple[Float[Tensor, "n_batch *shape_x"],
               Float[Tensor, "n_batch *shape_x"]
    ]:
        ...

    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"],
               z: Float[Tensor, "n_batch dim_z"]
    ) -> Float[Tensor, "..."]:
        ...

class VarEncoderNoPrior(nn.Module):
    """Variational encoder for the latent state of a dynamical system"""

    _empty_tensor: Tensor  # empty tensor to get device

    def __init__(
        self,
        config: VarAutoencoderConfig,
        encode_mean_net: nn.Module,
        encode_var_net: nn.Module
    ):
        super().__init__()
        self.config = config
        self.register_buffer("_empty_tensor", torch.empty(0))

        self.dim_mu = self.config.dim_mu
        self.shape_x = self.config.shape_x
        self.dim_z = self.config.dim_z
        self.n_win = self.config.n_win

        self.encode_mean = encode_mean_net
        self.encode_var = encode_var_net

        # check encoder mean network dims
        check_nn_dims(self.encode_mean,
                      ((self.dim_mu,), (self.n_win, *self.shape_x)),
                      ((self.dim_z,),),
                      "Encoder mean")

        # check encoder var network dims
        check_nn_dims(self.encode_var,
                      ((self.dim_mu,), (self.n_win, *self.shape_x)),
                      ((self.dim_z,),),
                      "Encoder variance")

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
                x_win: Float[Tensor, "n_batch n_win *shape_x"]
    ) -> tuple[Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"]
    ]:
        '''
        Get mean and variance of the latent state given the input data
        '''
        for p in self.parameters():
            assert torch.all(torch.isfinite(p)), f"Encoder: Parameter is not finite. Value: {p}"

        z_mean = self.encode_mean(mu, x_win)
        z_var = self.encode_var(mu, x_win)
        assert torch.all(z_var > 0), f"Encoder: Variance must be positive. Value: {z_var}"

        return z_mean, z_var

    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"],
               x_win: Float[Tensor, "n_batch n_win *shape_x"]
    ) -> Float[Tensor, "... dim_z"]:
        '''
        Get samples from the latent state distribution given the input data
        '''

        z_mean, z_var = self.forward(mu, x_win)
        z_mean = z_mean.unsqueeze(-2)
        z_stdev = torch.sqrt(z_var).unsqueeze(-2)

        n_batch = z_mean.shape[0]
        stdnorm_samples = torch.randn(n_batch, n_samples, self.dim_z, device=self.device)
        z = (z_mean + torch.mul(z_stdev, stdnorm_samples)).flatten(0, 1)

        # return samples with shape (n_batch*n_samples, dim_z)
        return z


class VarDecoderNoPrior(nn.Module):
    """Variational decoder for the latent state of a dynamical system"""

    _empty_tensor: Tensor  # empty tensor to get device

    def __init__(
        self,
        config: VarAutoencoderConfig,
        decode_mean_net: nn.Module,
        decode_var_net: nn.Module
    ):
        super().__init__()
        self.config = config
        self.register_buffer("_empty_tensor", torch.empty(0))
        self.dim_mu = self.config.dim_mu
        self.shape_x = self.config.shape_x
        self.dim_z = self.config.dim_z

        self.decode_mean = decode_mean_net
        self.decode_var = decode_var_net

        # check decoder mean network dims
        check_nn_dims(self.decode_mean,
                      ((self.dim_mu,), (self.dim_z,)),
                      (self.shape_x,),
                      "Decoder mean")

        # check decoder var network dims
        check_nn_dims(self.decode_var,
                      ((self.dim_mu,), (self.dim_z,)),
                      (self.shape_x,),
                      "Decoder variance")

    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device
    
    def resample_params(self) -> None:
        pass

    def kl_divergence(self) -> Float[Tensor, ""]:
        return torch.tensor(0.0)

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                z: Float[Tensor, "n_batch dim_z"]
    ) -> tuple[Float[Tensor, "n_batch *shape_x"],
               Float[Tensor, "n_batch *shape_x"]
    ]:
        '''
        Get mean and variance of the decoded state given the latent state
        '''
        for p in self.parameters():
            assert torch.all(torch.isfinite(p)), f"Decoder: Parameter is not finite. Value: {p}"

        x_mean = self.decode_mean(mu, z)
        x_var = self.decode_var(mu, z)
        assert torch.all(x_var > 0), f"Decoder: Variance must be positive. Value: {x_var} \n Min: {x_var.min()}"

        return x_mean, x_var

    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"],
               z: Float[Tensor, "n_batch dim_z"]
    ) -> Float[Tensor, "..."]:
        '''
        Get samples from the decoded state distribution given the latent state
        '''

        x_mean, x_var = self.forward(mu, z)
        n_x_dims = len(self.shape_x)

        x_mean = x_mean.unsqueeze(-n_x_dims-1)
        x_stdev = torch.sqrt(x_var).unsqueeze(-n_x_dims-1)

        n_batch = x_mean.shape[0]
        stdnorm_samples = torch.randn(n_batch, n_samples, *self.shape_x, device=self.device)
        x = (x_mean + torch.mul(x_stdev, stdnorm_samples)).flatten(0, 1)
        
        # return samples with shape (n_batch*n_samples, *shape_x)
        return x

