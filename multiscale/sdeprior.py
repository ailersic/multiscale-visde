import torch
from torch import nn, Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype

import visde
from visde.utils import check_nn_dims
# ruff: noqa: F821, F722

class MultiscaleLatentDriftNoPrior(nn.Module):
    """Multiscale latent drift with no prior on parameters"""

    _empty_tensor: Tensor  # empty tensor to get device

    def __init__(self,
                 config: visde.LatentDriftConfig,
                 dim_z_macro: int,
                 dim_z_micro: int,
                 macro_drift_net: nn.Module,
                 micro_drift_net: nn.Module,
                 macro_activ_net: nn.Module,
                 micro_activ_net: nn.Module
    ):
        super().__init__()
        self.register_buffer("_empty_tensor", torch.empty(0))

        self.config = config
        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro

        self.macro_drift_net = macro_drift_net
        self.micro_drift_net = micro_drift_net
        self.macro_activ_net = macro_activ_net
        self.micro_activ_net = micro_activ_net
        
        n_batch = 42
        test_inputs = [torch.zeros(n_batch, *shape) for shape in ((self.config.dim_mu,), (1,), (dim_z_macro,), (self.config.dim_f,))]
        test_output = self.macro_activ_net(*test_inputs)
        macro_activ_output_shape = test_output.shape[1:]

        test_inputs = [torch.zeros(n_batch, *shape) for shape in ((self.config.dim_mu,), (1,), (dim_z_micro,), (self.config.dim_f,))]
        test_output = self.micro_activ_net(*test_inputs)
        micro_activ_output_shape = test_output.shape[1:]

        # check drift network dims
        check_nn_dims(self.macro_drift_net,
                      ((self.config.dim_mu,), (1,), (dim_z_macro,), micro_activ_output_shape, (self.config.dim_f,)),
                      ((dim_z_macro,),),
                      "Macroscale latent drift")
        
        check_nn_dims(self.micro_drift_net,
                      ((self.config.dim_mu,), (1,), (dim_z_micro,), macro_activ_output_shape, (self.config.dim_f,)),
                      ((dim_z_micro,),),
                      "Microscale latent drift")
    
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    def resample_params(self) -> None:
        pass
    
    def kl_divergence(self) -> Tensor:
        return torch.tensor(0.0)
    
    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "batch dim_mu"],
                t: Float[Tensor, "batch 1"],
                z: Float[Tensor, "batch dim_z"],
                f: Float[Tensor, "batch dim_f"]
    ) -> Float[Tensor, "batch dim_z"]:
        """Compute the drift of the latent state z at time t given mu, f, and z"""

        if self.dim_z_micro > 0:
            z_macro = z[:, :self.dim_z_macro]
            z_micro = z[:, self.dim_z_macro:]

            macro_activ = self.macro_activ_net(mu, t, z_macro, f)
            micro_activ = self.micro_activ_net(mu, t, z_micro, f)

            macro_drift = self.macro_drift_net(mu, t, z_macro, micro_activ, f)
            micro_drift = self.micro_drift_net(mu, t, z_micro, macro_activ, f)
            drift = torch.cat((macro_drift, micro_drift), dim=1)
        else:
            z_macro = z
            micro_activ = torch.zeros(mu.shape[0], self.dim_z_micro, device=self.device)
            drift = self.macro_drift_net(mu, t, z_macro, micro_activ, f)
        
        return drift

class MultiscaleLatentDispersionNoPrior(nn.Module):
    """Multiscale latent dispersion with no prior on parameters"""

    _empty_tensor: Tensor  # empty tensor to get device

    def __init__(self,
                 config: visde.LatentDispersionConfig,
                 dim_z_macro: int,
                 dim_z_micro: int,
                 macro_net: nn.Module,
                 micro_net: nn.Module
    ):
        super().__init__()
        self.register_buffer("_empty_tensor", torch.empty(0))

        self.config = config
        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro

        self.macro_net = macro_net
        self.micro_net = micro_net
        
        # check dispersion network dims
        check_nn_dims(self.macro_net,
                      ((self.config.dim_mu,), (1,)),
                      ((self.dim_z_macro,),),
                      "Macroscale latent dispersion")

        check_nn_dims(self.micro_net,
                      ((self.config.dim_mu,), (1,)),
                      ((self.dim_z_micro,),),
                      "Microscale latent dispersion")
         
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device

    def resample_params(self) -> None:
        pass
    
    def kl_divergence(self) -> Tensor:
        return torch.tensor(0.0)
    
    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "batch dim_mu"],
                t: Float[Tensor, "batch 1"]
    ) -> Float[Tensor, "batch dim_z"]:
        """Compute the dispersion of the latent state z at time t given mu"""

        if self.dim_z_micro > 0:
            macro_dispersion = self.macro_net(mu, t)
            micro_dispersion = self.micro_net(mu, t)

            dispersion = torch.cat((macro_dispersion, micro_dispersion), dim=1)
        else:
            dispersion = self.macro_net(mu, t)
        
        return dispersion
