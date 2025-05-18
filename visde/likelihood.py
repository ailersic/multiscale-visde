import torch
from torch import nn, Tensor
from jaxtyping import Float, jaxtyped
from typing import Protocol, runtime_checkable
from beartype import beartype
# ruff: noqa: F821, F722

@runtime_checkable
class LogLike(Protocol):
    @jaxtyped(typechecker=beartype)
    def __call__(self,
                 x_true: Float[Tensor, "n_batch dim_x"],
                 x_mean: Float[Tensor, "n_batch n_samples dim_x"],
                 x_var: Float[Tensor, "n_batch n_samples dim_x"]
    ) -> Float[Tensor, "n_batch n_samples"]:
        ...

class LogLikeGaussian(nn.Module):
    _empty_tensor: Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("_empty_tensor", torch.empty(0))
    
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    @jaxtyped(typechecker=beartype)
    def forward(self,
                x_true: Float[Tensor, "n_batch dim_x"],
                x_mean: Float[Tensor, "n_batch n_samples dim_x"],
                x_var: Float[Tensor, "n_batch n_samples dim_x"],
    ) -> Float[Tensor, "n_batch n_samples"]:
        dim = x_true.shape[-1]
        klog2pi = torch.log(torch.tensor(2 * torch.pi, device=self.device)) * dim
        logdet = torch.log(x_var).sum(-1)

        x_diff = x_true.unsqueeze(1) - x_mean
        assert x_diff.shape == (x_true.shape[0], x_mean.shape[1], dim)

        sqnorm = x_diff.pow(2).div(x_var).sum(-1)
        loglike = -0.5 * (klog2pi + logdet + sqnorm)

        return loglike