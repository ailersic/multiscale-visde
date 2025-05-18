import torch
from torch import nn, Tensor
from typing import Protocol
from jaxtyping import Float, Union
# ruff: noqa: F821, F722

def softplus(x: Tensor) -> Tensor:
    return torch.log(1 + torch.exp(x))

def inverse_softplus(x: Tensor) -> Tensor:
    return torch.log(torch.exp(x) - 1)

class Kernel(Protocol):
    def __call__(
        self,
        t1: Float[Tensor, "n 1"],
        t2: Float[Tensor, "m 1"]
    ) -> Float[Tensor, "n m"]:
        ...

    @property
    def var(self) -> Tensor:
        ...

class DeepGaussianKernel(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        batch_size: int,
        dt: Union[float, Tensor],
        len_init: Union[float, Tensor] = 1e-1,
        var_init: Union[float, Tensor] = 1e-2,
    ) -> None:
        super().__init__()
        self.net = net
        self.rawsigma = nn.Parameter(inverse_softplus(torch.tensor(1.0)))

        if isinstance(len_init, Tensor):
            self.rawlens = nn.Parameter(inverse_softplus(len_init.clone().detach()))
        else:
            self.rawlens = nn.Parameter(inverse_softplus(torch.tensor(len_init)))
        
        if isinstance(var_init, Tensor):
            self.rawvar = nn.Parameter(inverse_softplus(var_init.clone().detach()))
        else:
            self.rawvar = nn.Parameter(inverse_softplus(torch.tensor(var_init)))

        if isinstance(dt, Tensor):
            self.register_buffer("offset", torch.tensor(0.5 * (batch_size - 1) * dt.item()))
            self.register_buffer("scale", torch.tensor(0.5 * (batch_size - 1) * dt.item()))
        else:
            self.register_buffer("offset", torch.tensor(0.5 * (batch_size - 1) * dt))
            self.register_buffer("scale", torch.tensor(0.5 * (batch_size - 1) * dt))

    @property
    def lens(self) -> Float[Tensor, ""]:
        return softplus(self.rawlens)
    
    @property
    def var(self) -> Float[Tensor, ""]:
        return softplus(self.rawvar)

    @property
    def sigma(self) -> Float[Tensor, ""]:
        return softplus(self.rawsigma)

    def rescale(self,
                t: Float[Tensor, "n 1"]
    ) -> Float[Tensor, "n 1"]:
        return (t - self.offset) / self.scale

    def forward(self,
                t1: Float[Tensor, "n 1"],
                t2: Float[Tensor, "m 1"]
    ) -> Float[Tensor, "m n"]:
        t1, t2 = self.rescale(t1), self.rescale(t2)
        t1, t2 = self.net(t1) + t1, self.net(t2) + t2

        t_diff = t1.view(t1.shape[0], 1, 1) - t2.view(1, t2.shape[0], 1)
        l2 = self.lens.pow(2)
        return torch.exp(-0.5*t_diff.pow(2).div(l2).sum(-1)).mul(self.sigma)