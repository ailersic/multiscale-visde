import torch
#import torchvision
from torch import Tensor, nn
from torch.optim import lr_scheduler
from jaxtyping import Float, jaxtyped
from beartype import beartype
#import matplotlib.pyplot as plt

import pickle as pkl
import numpy as np
import os
import pathlib
import math

import visde
# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def softplus(x: Tensor) -> Tensor:
    return torch.log(1 + torch.exp(x))

class EncodeMeanNet(nn.Module):
    def __init__(self, config, shape_z, dim_z_macro, dim_z_micro, shared_kernel, shared_bias, n_sigma):
        super(EncodeMeanNet, self).__init__()
        n_win = 1#config.n_win

        self.dim_x = config.dim_x
        self.shape_x = config.shape_x
        n_chan = self.shape_x[0]

        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro
        self.sigma = int(np.sqrt(self.dim_x//self.dim_z_macro) + 0.5)
        print(f"Sigma: {self.sigma}")

        self.smooth_net = nn.Sequential(nn.Unflatten(1, self.shape_x),
                                        nn.Conv2d(n_win*n_chan, n_chan, kernel_size=2*n_sigma*self.sigma + 1, stride=1,
                                                  padding_mode="circular", padding=n_sigma*self.sigma, groups=n_chan),
                                        nn.Flatten())

        self.macro_net = nn.Sequential(nn.Unflatten(1, self.shape_x),
                                       nn.Conv2d(n_win*n_chan, n_chan, kernel_size=2*n_sigma*self.sigma + 1,
                                                 stride=self.sigma, padding_mode="circular", padding=n_sigma*self.sigma, groups=n_chan),
                                       nn.Flatten())

        self.smooth_net[1].weight = shared_kernel
        self.smooth_net[1].bias = shared_bias
        self.macro_net[1].weight = shared_kernel
        self.macro_net[1].bias = shared_bias

        micro_padding = 4
        micro_kernel = 2*micro_padding + 1

        if self.dim_z_micro > 0:
            self.micro_net = nn.Sequential(nn.Unflatten(1, self.shape_x),
                                           nn.Conv2d(n_win*n_chan, 16, kernel_size=micro_kernel, stride=4, padding=micro_padding, padding_mode="circular"),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(16, 32, kernel_size=micro_kernel, stride=4, padding=micro_padding, padding_mode="circular"),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(32, 64, kernel_size=micro_kernel, stride=4, padding=micro_padding, padding_mode="circular"),
                                           nn.LeakyReLU(),
                                           nn.Flatten(),
                                           nn.Linear(64*5*2, self.dim_z_micro)
                                           )

            for layer in self.micro_net:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        n_batch = x_win.shape[0]
        x = x_win.flatten(1)

        z_macro = self.macro_net(x)
        x_smooth = self.smooth_net(x)
        x_delta = x - x_smooth

        if self.dim_z_micro == 0:
            z_micro = torch.zeros(n_batch, self.dim_z_micro, device=z_macro.device)
        else:
            z_micro = self.micro_net(x_delta)

        return torch.cat([z_macro, z_micro], dim=-1)

class EncodeVarNet(nn.Module):
    def __init__(self, config, shape_z, dim_z_macro, dim_z_micro):
        super(EncodeVarNet, self).__init__()
        dim_z = config.dim_z

        self.out_activ = nn.Softplus()
        self.fixed_var = nn.Parameter(-4*torch.ones((1, dim_z)))

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        z_var_norm = self.fixed_var.expand(x_win.shape[0], *self.fixed_var.shape[1:])
        return self.out_activ(z_var_norm)

class DecodeMeanNet(nn.Module):
    def __init__(self, config, shape_z, dim_z_macro, dim_z_micro, shared_kernel, shared_bias, n_sigma, _macro_var, _micro_var):
        super(DecodeMeanNet, self).__init__()

        self.dim_x = config.dim_x
        self.shape_x = config.shape_x
        n_chan = self.shape_x[0]

        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro
        self.sigma = int(np.sqrt(self.dim_x//self.dim_z_macro))
        self.shape_z = shape_z

        self.macro_net = nn.Sequential(nn.Unflatten(1, self.shape_z),
                                       nn.ConvTranspose2d(n_chan, n_chan, kernel_size=2*n_sigma*self.sigma + 1, stride=self.sigma,
                                                          padding=n_sigma*self.sigma, output_padding=self.sigma-1, groups=n_chan),
                                       nn.Flatten())

        self.macro_net[1].weight = shared_kernel
        self.macro_net[1].bias = shared_bias

        micro_padding = 4
        micro_kernel = 2*micro_padding + 1

        self.out_activ = nn.Softplus()
        self._macro_var = _macro_var
        self._micro_var = _micro_var

        if self.dim_z_micro > 0:
            self.micro_net = nn.Sequential(nn.Linear(self.dim_z_micro, 64*5*2),
                                           nn.Unflatten(1, (64, 5, 2)),
                                           nn.LeakyReLU(),
                                           nn.ConvTranspose2d(64, 32, kernel_size=micro_kernel, stride=4, padding=micro_padding, output_padding=(3,0)),
                                           nn.LeakyReLU(),
                                           nn.ConvTranspose2d(32, 16, kernel_size=micro_kernel, stride=4, padding=micro_padding, output_padding=(3,3)),
                                           nn.LeakyReLU(),
                                           nn.ConvTranspose2d(16, n_chan, kernel_size=micro_kernel, stride=4, padding=micro_padding, output_padding=(3,3)),
                                           nn.Flatten()
                                           )

            for layer in self.micro_net:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def decode_macro(self, z_macro: Tensor) -> Tensor:
        return self.macro_net(z_macro)

    def decode_micro(self, z_micro: Tensor) -> Tensor:
        n_batch = z_micro.shape[0]

        if self.dim_z_micro == 0:
            x_micro = torch.zeros(n_batch, self.dim_x, device=z_micro.device)
        else:
            x_micro_unnorm = self.micro_net(z_micro).unflatten(1, self.shape_x)

            batch_macro_var = self.macro_var.expand(n_batch, *self.shape_x)
            batch_micro_var = self.micro_var.expand(n_batch, *self.shape_x)
            joint_var = (batch_macro_var.pow(-1) + batch_micro_var.pow(-1)).pow(-1)

            x_micro = x_micro_unnorm.mul(joint_var).div(batch_micro_var).flatten(1)
        return x_micro
    
    @property
    def macro_var(self) -> Tensor:
        return self.out_activ(self._macro_var)
    
    @property
    def micro_var(self) -> Tensor:
        return self.out_activ(self._micro_var)

    def forward(self, mu: Tensor, z: Tensor) -> Tensor:
        z_macro = z[:, :self.dim_z_macro]
        z_micro = z[:, self.dim_z_macro:]

        x_macro = self.decode_macro(z_macro).unflatten(1, self.shape_x)
        x_micro = self.decode_micro(z_micro).unflatten(1, self.shape_x)

        x = x_macro + x_micro

        return x

class DecodeVarNet(nn.Module):
    def __init__(self, config, shape_z, dim_z_macro, dim_z_micro, _macro_var, _micro_var):
        super(DecodeVarNet, self).__init__()
        self.shape_x = config.shape_x
        
        self.out_activ = nn.Softplus()
        self._macro_var = _macro_var
        self._micro_var = _micro_var
    
    @property
    def macro_var(self) -> Tensor:
        return self.out_activ(self._macro_var)
    
    @property
    def micro_var(self) -> Tensor:
        return self.out_activ(self._micro_var)

    def forward(self, mu: Tensor, z: Tensor) -> Tensor:
        joint_var = (self.macro_var.pow(-1) + self.micro_var.pow(-1)).pow(-1)
        return joint_var.expand(z.shape[0], *self.shape_x)

class DriftNet(nn.Module):
    def __init__(self, config, shape_z, dim_z_macro, dim_z_micro):
        super(DriftNet, self).__init__()
        self.dim_z = config.dim_z
        self.shape_z = shape_z
        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro

        self.r = 2
        dim_z_macro_activ = dim_z_micro
        dim_z_grid = int(np.prod(self.shape_z[1:]))

        _adj = torch.zeros((*self.shape_z[1:], *self.shape_z[1:])).to(torch.bool)
        for i in range(self.shape_z[1]):
            for j in range(self.shape_z[2]):
                for ir in range(i - self.r, i + self.r + 1):
                    for jr in range(j - self.r, j + self.r + 1):
                        _adj[i, j, ir % self.shape_z[1], jr % self.shape_z[2]] = True
        _adj = _adj.reshape(dim_z_grid, dim_z_grid)

        self.adj = _adj.nonzero(as_tuple=True)[1].reshape(dim_z_grid, (2*self.r + 1)**2)

        self.macro_net = nn.Sequential(nn.Linear(self.shape_z[0]*(2*self.r + 1)**2 + self.dim_z_micro, 128),
                                        nn.LeakyReLU(),
                                        nn.Linear(128, 512),
                                        nn.LeakyReLU(),
                                        nn.Linear(512, 128),
                                        nn.LeakyReLU(),
                                        nn.Linear(128, self.shape_z[0]))
        self.macro_vmap = torch.vmap(self.macro_drift, in_dims=(2, 1), out_dims=2)

        self.micro_net = nn.Sequential(nn.Linear(dim_z_macro_activ + self.dim_z_micro, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, self.dim_z_micro))

        self.micro_activ_net = nn.Sequential(nn.Linear(self.dim_z_micro + 4, 128),
                                             nn.ReLU(),
                                             nn.Linear(128, 128),
                                             nn.ReLU(),
                                             nn.Linear(128, self.dim_z_micro))
        self.macro_activ = nn.Linear(self.dim_z_macro, dim_z_macro_activ)
        
        for layer in self.macro_net:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.micro_net:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def macro_drift(self,
                    z_macro_adj: Float[Tensor, "n_batch n_chan dim_z_macro"],
                    z_micro: Float[Tensor, "n_batch dim_z_micro"]
    ) -> Float[Tensor, "n_batch n_chan"]:
        dz = self.macro_net(torch.cat([z_macro_adj.flatten(1), z_micro], dim=-1))
        return dz

    def micro_activ(self,
                    z_micro: Float[Tensor, "n_batch dim_z_micro"],
                    t: Float[Tensor, "n_batch 1"],
                    ij: Float[Tensor, "n_batch 2"]
    ) -> Float[Tensor, "n_batch dim_z_micro"]:
        z_micro_activ = z_micro + self.micro_activ_net(torch.cat([z_micro, torch.cos(t), torch.sin(t), ij], dim=-1))
        return z_micro_activ

    def forward(self,
                 mu: Float[Tensor, "n_batch dim_mu"],
                 t: Float[Tensor, "n_batch 1"],
                 z: Float[Tensor, "n_batch dim_z"],
                 f: Float[Tensor, "n_batch dim_f"]
    ) -> Float[Tensor, "n_batch dim_z"]:
        z_macro = z[:, 0:self.dim_z_macro]
        z_micro = z[:, self.dim_z_macro:]
        n_batch = z.shape[0]

        dim_z_grid = int(np.prod(self.shape_z[1:]))
        z_macro_reshaped = z_macro.reshape(n_batch, self.shape_z[0], dim_z_grid)[:, :, self.adj]

        z_micro_reshaped = z_micro.unsqueeze(1).expand(-1, dim_z_grid, -1).flatten(0, 1)
        t_reshaped = t.unsqueeze(1).expand(-1, dim_z_grid, -1).flatten(0, 1)
        ij = torch.stack(torch.meshgrid(torch.arange(self.shape_z[1]), torch.arange(self.shape_z[2])), dim=-1).to(z.device).flatten(0, 1)
        ij_reshaped = ij.unsqueeze(0).expand(n_batch, -1, -1).flatten(0, 1)
        z_micro_activ = self.micro_activ(z_micro_reshaped, t_reshaped, ij_reshaped).unflatten(0, (n_batch, dim_z_grid))

        dzdt_macro = self.macro_vmap(z_macro_reshaped, z_micro_activ).flatten(1)
        dzdt_micro = self.micro_net(torch.cat([z_micro, self.macro_activ(z_macro)], dim=-1))

        return torch.cat([dzdt_macro, dzdt_micro], dim=-1)

class DispNet(nn.Module):
    def __init__(self, config, dim_z_macro, dim_z_micro):
        super(DispNet, self).__init__()
        dim_z = config.dim_z
        self.out_activ = nn.Softplus()
        self.fixed_disp = nn.Parameter(torch.ones((1, dim_z)))

    def forward(self, mu: Tensor, t: Tensor) -> Tensor:
        disp_norm = self.fixed_disp.expand(mu.shape[0], *self.fixed_disp.shape[1:])
        return self.out_activ(disp_norm)

class KernelNet(nn.Module):
    def __init__(self, config):
        super(KernelNet, self).__init__()

        self.net = nn.Sequential(nn.Linear(1, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 1))
    
    def forward(self, t: Tensor) -> Tensor:
        return self.net(t)

def create_latent_sde(dim_z_macro: int,
                      dim_z_micro: int,
                      n_batch: int,
                      n_win: int,
                      lr: float,
                      lr_sched_freq: int,
                      data_file: str,
                      device: torch.device = torch.device("cuda:0")
) -> visde.sde.LatentSDE:
    with open(os.path.join(CURR_DIR, data_file), "rb") as f:
        data = pkl.load(f)
    
    dim_mu = data["train_mu"].shape[1]
    shape_x = tuple(data["train_x"].shape[2:])
    dim_x = int(np.prod(shape_x))
    dim_z = dim_z_macro + dim_z_micro
    dim_f = data["train_f"].shape[2]
    #dt = data["train_t"][0,1] - data["train_t"][0,0]

    sigma = int(np.sqrt(dim_x//dim_z_macro))
    shape_z = (shape_x[0], *[grid_ax // sigma for grid_ax in shape_x[1:]])
    print(shape_z)

    n_sigma = 2
    n_chan = shape_x[0]

    #shared_kernel = nn.Parameter((2*torch.rand(n_chan, 1, 2*n_sigma*sigma + 1, 2*n_sigma*sigma + 1) - 1)/
    #                                np.sqrt(n_chan*(2*n_sigma*sigma + 1)*(2*n_sigma*sigma + 1)))
    kernel_range_x = torch.arange(-n_sigma*sigma, n_sigma*sigma + 1).view(1, 1, -1, 1).tile((n_chan, 1, 1, 1))
    kernel_range_y = torch.arange(-n_sigma*sigma, n_sigma*sigma + 1).view(1, 1, 1, -1).tile((n_chan, 1, 1, 1))
    shared_kernel = nn.Parameter(torch.exp(-(kernel_range_x**2 + kernel_range_y**2)/(2*sigma**2))/
                                np.sqrt(n_chan*(2*n_sigma*sigma + 1)*(2*n_sigma*sigma + 1)))
    shared_bias = nn.Parameter(torch.zeros(n_chan))

    # encoder
    vaeconfig = visde.VarAutoencoderConfig(dim_mu=dim_mu, dim_x=dim_x, dim_z=dim_z, n_win=n_win, shape_x=shape_x)
    encode_mean_net = EncodeMeanNet(vaeconfig, shape_z, dim_z_macro, dim_z_micro, shared_kernel, shared_bias, n_sigma)
    encode_var_net = EncodeVarNet(vaeconfig, shape_z, dim_z_macro, dim_z_micro)
    encoder = visde.VarEncoderNoPrior(vaeconfig, encode_mean_net, encode_var_net)

    _macro_var = nn.Parameter(-4*torch.ones((1, *shape_x)))
    _micro_var = nn.Parameter(-4*torch.ones((1, *shape_x)))

    # decoder        
    decode_mean_net = DecodeMeanNet(vaeconfig, shape_z, dim_z_macro, dim_z_micro, shared_kernel, shared_bias, n_sigma, _macro_var, _micro_var)
    decode_var_net = DecodeVarNet(vaeconfig, shape_z, dim_z_macro, dim_z_micro, _macro_var, _micro_var)
    decoder = visde.VarDecoderNoPrior(vaeconfig, decode_mean_net, decode_var_net)

    # drift
    config = visde.LatentDriftConfig(dim_mu=dim_mu, dim_z=dim_z, dim_f=dim_f)
    driftnet = DriftNet(config, shape_z, dim_z_macro, dim_z_micro)
    drift = visde.LatentDriftNoPrior(config, driftnet)

    # dispersion
    config = visde.LatentDispersionConfig(dim_mu=dim_mu, dim_z=dim_z)
    dispnet = DispNet(config, dim_z_macro, dim_z_micro)
    dispersion = visde.LatentDispersionNoPrior(config, dispnet)

    # likelihood
    loglikelihood = visde.LogLikeGaussian()

    # latent variational distribution
    config = visde.LatentVarConfig(dim_mu=dim_mu, dim_z=dim_z)
    #kernel_net = KernelNet(config)
    #kernel = visde.DeepGaussianKernel(kernel_net, n_batch, dt)
    #latentvar = visde.AmortizedLatentVarGP(config, kernel, encoder)
    latentvar = visde.ParamFreeLatentVarGP(config, encoder)

    config = visde.LatentSDEConfig(n_totaldata=torch.numel(data["train_t"]),
                                   n_samples=1,
                                   n_tquad=0,
                                   n_warmup=0,
                                   n_transition=5000,
                                   lr=lr,
                                   lr_sched_freq=lr_sched_freq)
    model = visde.LatentSDE(config=config,
                            encoder=encoder,
                            decoder=decoder,
                            drift=drift,
                            dispersion=dispersion,
                            loglikelihood=loglikelihood,
                            latentvar=latentvar).to(device)
    
    return model

