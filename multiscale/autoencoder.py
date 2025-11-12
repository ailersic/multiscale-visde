import torch
from torch import Tensor, nn
from jaxtyping import Float, jaxtyped
from beartype import beartype

import visde
from visde.utils import check_nn_dims
# ruff: noqa: F821, F722

class MultiscaleVarEncoderNoPrior(nn.Module):
    """Variational encoder for a multiscale latent state with no prior on parameters"""
    
    _empty_tensor: Tensor  # empty tensor to get device

    def __init__(self,
                 config: visde.VarAutoencoderConfig,
                 dim_z_macro: int,
                 dim_z_micro: int,
                 macro_mean_net: nn.Module,
                 macro_var_net: nn.Module,
                 micro_mean_net: nn.Module,
                 micro_var_net: nn.Module,
                 smooth_net: nn.Module
    ):
        super().__init__()
        self.register_buffer("_empty_tensor", torch.empty(0))

        self.shape_x = config.shape_x
        
        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro
        self.dim_z = dim_z_macro + dim_z_micro

        self.macro_mean_net = macro_mean_net
        self.macro_var_net = macro_var_net

        self.micro_mean_net = micro_mean_net
        self.micro_var_net = micro_var_net

        self.smooth_net = smooth_net

        # check macro mean network dims
        check_nn_dims(self.macro_mean_net,
                      ((config.dim_mu,), (config.n_win, *self.shape_x)),
                      ((self.dim_z_macro,),),
                      "Macro encoder mean")
        
        # check macro var network dims
        check_nn_dims(self.macro_var_net,
                      ((config.dim_mu,), (config.n_win, *self.shape_x)),
                      ((self.dim_z_macro,),),
                      "Macro encoder variance")
        
        # check micro mean network dims
        check_nn_dims(self.micro_mean_net,
                      ((config.dim_mu,), (config.n_win, *self.shape_x)),
                      ((self.dim_z_micro,),),
                      "Micro encoder mean")
        
        # check micro var network dims
        check_nn_dims(self.micro_var_net,
                      ((config.dim_mu,), (config.n_win, *self.shape_x)),
                      ((self.dim_z_micro,),),
                      "Micro encoder variance")
        
        # check smooth network dims
        check_nn_dims(self.smooth_net,
                      ((config.dim_mu,), (config.n_win, *self.shape_x)),
                      ((config.n_win, *self.shape_x),),
                      "Smooth network")
    
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
                x_win: Float[Tensor, "n_batch n_win *shape_x"]
    ) -> tuple[Float[Tensor, "n_batch dim_z"],
               Float[Tensor, "n_batch dim_z"]
    ]:
        n_batch = x_win.shape[0]

        z_macro_mean = self.macro_mean_net(mu, x_win)
        z_macro_var = self.macro_var_net(mu, x_win)

        x_smooth = self.smooth_net(mu, x_win)
        x_delta = x_win - x_smooth

        if self.dim_z_micro == 0:
            z_micro_mean = torch.zeros(n_batch, self.dim_z_micro, device=self.device)
            z_micro_var = torch.zeros(n_batch, self.dim_z_micro, device=self.device)
        else:
            z_micro_mean = self.micro_mean_net(mu, x_delta)
            z_micro_var = self.micro_var_net(mu, x_delta)

        z_mean = torch.cat([z_macro_mean, z_micro_mean], dim=-1)
        z_var = torch.cat([z_macro_var, z_micro_var], dim=-1)

        return z_mean, z_var
    
    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"],
               x_win: Float[Tensor, "n_batch n_win *shape_x"]
    ) -> Float[Tensor, "... dim_z"]:
        z_mean, z_var = self.forward(mu, x_win)
        z_mean = z_mean.unsqueeze(-2)
        z_stdev = torch.sqrt(z_var).unsqueeze(-2)

        n_batch = z_mean.shape[0]
        stdnorm_samples = torch.randn(n_batch, n_samples, self.dim_z, device=self.device)
        z = (z_mean + torch.mul(z_stdev, stdnorm_samples)).flatten(0, 1)
        
        return z

class MultiscaleVarDecoderNoPrior(nn.Module):
    """Variational decoder for a multiscale latent state with no prior on parameters"""
    
    _empty_tensor: Tensor  # empty tensor to get device

    def __init__(self,
                 config: visde.VarAutoencoderConfig,
                 dim_z_macro: int,
                 dim_z_micro: int,
                 macro_mean_net: nn.Module,
                 macro_var_net: nn.Module,
                 micro_mean_net: nn.Module,
                 micro_var_net: nn.Module
    ):
        super().__init__()
        self.register_buffer("_empty_tensor", torch.empty(0))

        self.shape_x = config.shape_x
        
        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro

        self.macro_mean_net = macro_mean_net
        self.macro_var_net = macro_var_net

        self.micro_mean_net = micro_mean_net
        self.micro_var_net = micro_var_net

        # check macro mean network dims
        check_nn_dims(self.macro_mean_net,
                      ((config.dim_mu,), (self.dim_z_macro,)),
                      (self.shape_x,),
                      "Macro decoder mean")
        
        # check macro var network dims
        check_nn_dims(self.macro_var_net,
                      ((config.dim_mu,), (self.dim_z_macro,)),
                      (self.shape_x,),
                      "Macro decoder variance")
        
        if dim_z_micro > 0:
            # check micro mean network dims
            check_nn_dims(self.micro_mean_net,
                         ((config.dim_mu,), (self.dim_z_micro,)),
                         (self.shape_x,),
                         "Micro decoder mean")
            
            # check micro var network dims
            check_nn_dims(self.micro_var_net,
                         ((config.dim_mu,), (self.dim_z_micro,)),
                         (self.shape_x,),
                         "Micro decoder variance")
    
    @property
    def device(self) -> torch.device:
        return self._empty_tensor.device
    
    def resample_params(self) -> None:
        pass

    def kl_divergence(self) -> Float[Tensor, ""]:
        return torch.tensor(0.0)
    
    @jaxtyped(typechecker=beartype)
    def macro_mean(self,
                   mu: Float[Tensor, "n_batch dim_mu"],
                   z_macro: Float[Tensor, "n_batch dim_z_macro"]
    ) -> Float[Tensor, "n_batch *shape_x"]:
        return self.macro_mean_net(mu, z_macro)
    
    @jaxtyped(typechecker=beartype)
    def micro_mean(self,
                   mu: Float[Tensor, "n_batch dim_mu"],
                   z_micro: Float[Tensor, "n_batch dim_z_micro"]
    ) -> Float[Tensor, "n_batch *shape_x"]:
        x_micro_unnorm = self.micro_mean_net(mu, z_micro)
        x_micro_var = self.micro_var_net(mu, z_micro)

        x_macro_var = self.macro_var_net(mu, z_micro)
        x_var = (x_macro_var.pow(-1) + x_micro_var.pow(-1)).pow(-1)

        x_micro_mean = x_micro_unnorm.mul(x_var).div(x_micro_var)
        return x_micro_mean

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                z: Float[Tensor, "n_batch dim_z"]
    ) -> tuple[Float[Tensor, "n_batch *shape_x"],
               Float[Tensor, "n_batch *shape_x"]
    ]:
        z_macro = z[:, :self.dim_z_macro]
        z_micro = z[:, self.dim_z_macro:]

        x_macro_mean = self.macro_mean_net(mu, z_macro)
        x_macro_var = self.macro_var_net(mu, z_macro)
        
        if self.dim_z_micro == 0:
            x_micro_mean = torch.zeros_like(x_macro_mean, device=self.device)
            x_var = x_macro_var
        else:
            x_micro_unnorm = self.micro_mean_net(mu, z_micro)
            x_micro_var = self.micro_var_net(mu, z_micro)

            x_var = (x_macro_var.pow(-1) + x_micro_var.pow(-1)).pow(-1)
            x_micro_mean = x_micro_unnorm.mul(x_var).div(x_micro_var)

        x_mean = x_macro_mean + x_micro_mean

        return x_mean, x_var

    @jaxtyped(typechecker=beartype)
    def sample(self,
               n_samples: int,
               mu: Float[Tensor, "n_batch dim_mu"],
               z: Float[Tensor, "n_batch dim_z"]
    ) -> Float[Tensor, "..."]:
        x_mean, x_var = self.forward(mu, z)
        n_x_dims = len(self.shape_x)

        x_mean = x_mean.unsqueeze(-n_x_dims-1)
        x_stdev = torch.sqrt(x_var).unsqueeze(-n_x_dims-1)

        n_batch = x_mean.shape[0]
        stdnorm_samples = torch.randn(n_batch, n_samples, *self.shape_x, device=self.device)
        x_samples = (x_mean + torch.mul(x_stdev, stdnorm_samples)).flatten(0, 1)
        
        return x_samples
