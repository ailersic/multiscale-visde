import torch
#import torchvision
from torch import Tensor, nn
from torch.optim import lr_scheduler
from jaxtyping import Float, jaxtyped
from beartype import beartype
from kolsol.torch.solver import KolSol

import numpy as np
import os
import pathlib
import math

import visde
import multiscale
# ruff: noqa: F821, F722

from experiments_refac.shallow_water_2d.utils import CaseConfig, get_version_str
from datasets.shallow_water_2d.load_data import full_data_path

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def softplus(x: Tensor) -> Tensor:
    return torch.log(1 + torch.exp(x))

class EncodeMeanMacroNet(nn.Module):
    def __init__(self, config, dim_z_macro, shared_kernel, shared_bias, n_sigma):
        super(EncodeMeanMacroNet, self).__init__()
        n_chan = config.shape_x[0]
        sigma = int(np.sqrt(config.dim_x//dim_z_macro) + 0.5)

        self.net = nn.Sequential(nn.Flatten(1, 2),
                                 nn.Conv2d(config.n_win*n_chan, n_chan, kernel_size=2*n_sigma*sigma + 1, stride=sigma,
                                           padding_mode="circular", padding=n_sigma*sigma, groups=n_chan),
                                 nn.Flatten())

        self.net[1].weight = shared_kernel
        self.net[1].bias = shared_bias

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        return self.net(x_win)

class EncodeSmoothNet(nn.Module):
    def __init__(self, config, dim_z_macro, shared_kernel, shared_bias, n_sigma):
        super(EncodeSmoothNet, self).__init__()
        n_chan = config.shape_x[0]
        sigma = int(np.sqrt(config.dim_x//dim_z_macro) + 0.5)

        self.net = nn.Sequential(nn.Flatten(1, 2),
                                 nn.Conv2d(config.n_win*n_chan, config.n_win*n_chan, kernel_size=2*n_sigma*sigma + 1, stride=1,
                                           padding_mode="circular", padding=n_sigma*sigma, groups=n_chan),
                                 nn.Unflatten(1, (config.n_win, n_chan)))

        self.net[1].weight = shared_kernel
        self.net[1].bias = shared_bias

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        return self.net(x_win)

class EncodeMeanMicroNet(nn.Module):
    def __init__(self, config, dim_z_micro):
        super(EncodeMeanMicroNet, self).__init__()
        n_chan = config.shape_x[0]

        micro_padding = 4
        micro_kernel = 2*micro_padding + 1
        self.dim_z_micro = dim_z_micro
        dim_hidden = 16 * config.shape_x[1] * config.shape_x[2] // (8 * 8)

        if dim_z_micro > 0:
            self.net = nn.Sequential(nn.Flatten(1, 2),
                                     nn.Conv2d(config.n_win*n_chan, 64, kernel_size=micro_kernel, stride=2, padding=micro_padding, padding_mode="circular"),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(64, 32, kernel_size=micro_kernel, stride=2, padding=micro_padding, padding_mode="circular"),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(32, 16, kernel_size=micro_kernel, stride=2, padding=micro_padding, padding_mode="circular"),
                                     nn.LeakyReLU(),
                                     nn.Flatten(),
                                     nn.Linear(dim_hidden, dim_z_micro)
                                     )

            for layer in self.net:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        if self.dim_z_micro == 0:
            return torch.zeros(x_win.shape[0], self.dim_z_micro, device=x_win.device)
        else:
            return self.net(x_win)

class EncodeVarMacroNet(nn.Module):
    def __init__(self, config, dim_z_macro):
        super(EncodeVarMacroNet, self).__init__()
        
        self.out_activ = nn.Softplus()
        self.fixed_var = nn.Parameter(-4*torch.ones((1, dim_z_macro)))

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        z_macro_var_norm = self.fixed_var.expand(x_win.shape[0], *self.fixed_var.shape[1:])
        return self.out_activ(z_macro_var_norm)

class EncodeVarMicroNet(nn.Module):
    def __init__(self, config, dim_z_micro):
        super(EncodeVarMicroNet, self).__init__()
        
        self.out_activ = nn.Softplus()
        self.fixed_var = nn.Parameter(-4*torch.ones((1, dim_z_micro)))

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        z_micro_var_norm = self.fixed_var.expand(x_win.shape[0], *self.fixed_var.shape[1:])
        return self.out_activ(z_micro_var_norm)

class DecodeMeanMacroNet(nn.Module):
    def __init__(self, config, shape_z, dim_z_macro, shared_kernel, shared_bias, n_sigma):
        super(DecodeMeanMacroNet, self).__init__()

        self.dim_x = config.dim_x
        self.shape_x = config.shape_x
        n_chan = self.shape_x[0]

        self.dim_z_macro = dim_z_macro
        self.sigma = int(np.sqrt(config.dim_x//dim_z_macro) + 0.5)
        self.n_sigma = n_sigma
        self.shape_z = shape_z

        self.net = nn.Sequential(nn.Unflatten(1, self.shape_z),
                                 nn.CircularPad2d(n_sigma),
                                 nn.ConvTranspose2d(n_chan, n_chan, kernel_size=2*n_sigma*self.sigma + 1, stride=self.sigma,
                                                    padding=n_sigma*self.sigma, output_padding=self.sigma-1, groups=n_chan))

        self.net[2].weight = shared_kernel
        self.net[2].bias = shared_bias

    def forward(self, mu: Tensor, z_macro: Tensor) -> Tensor:
        return self.net(z_macro)[:, :, self.n_sigma*self.sigma:-self.n_sigma*self.sigma, self.n_sigma*self.sigma:-self.n_sigma*self.sigma]

class DecodeMeanMicroNet(nn.Module):
    def __init__(self, config, dim_z_micro):
        super(DecodeMeanMicroNet, self).__init__()

        self.shape_x = config.shape_x
        n_chan = self.shape_x[0]

        micro_padding = 4
        micro_kernel = 2*micro_padding + 1
        dim_hidden = 16 * config.shape_x[1] * config.shape_x[2] // (8 * 8)

        self.dim_z_micro = dim_z_micro

        if dim_z_micro > 0:
            self.net = nn.Sequential(nn.Linear(dim_z_micro, dim_hidden),
                                     nn.LeakyReLU(),
                                     nn.Unflatten(1, (16, 16, 16)),
                                     nn.ConvTranspose2d(16, 32, kernel_size=micro_kernel, stride=2, padding=micro_padding, output_padding=(1,1)),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(32, 64, kernel_size=micro_kernel, stride=2, padding=micro_padding, output_padding=(1,1)),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(64, n_chan, kernel_size=micro_kernel, stride=2, padding=micro_padding, output_padding=(1,1)),
                                     )

            for layer in self.net:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, mu: Tensor, z_micro: Tensor) -> Tensor:
        if self.dim_z_micro == 0:
            return torch.zeros(z_micro.shape[0], self.shape_x, device=z_micro.device)
        else:
            return self.net(z_micro)

class DecodeVarMacroNet(nn.Module):
    def __init__(self, config):
        super(DecodeVarMacroNet, self).__init__()
        
        self.out_activ = nn.Softplus()
        self.fixed_var = nn.Parameter(-4*torch.ones((1, *config.shape_x)))

    def forward(self, mu: Tensor, z_macro: Tensor) -> Tensor:
        z_macro_var_norm = self.fixed_var.expand(z_macro.shape[0], *self.fixed_var.shape[1:])
        return self.out_activ(z_macro_var_norm)

class DecodeVarMicroNet(nn.Module):
    def __init__(self, config):
        super(DecodeVarMicroNet, self).__init__()
        
        self.out_activ = nn.Softplus()
        self.fixed_var = nn.Parameter(-4*torch.ones((1, *config.shape_x)))

    def forward(self, mu: Tensor, z_micro: Tensor) -> Tensor:
        z_micro_var_norm = self.fixed_var.expand(z_micro.shape[0], *self.fixed_var.shape[1:])
        return self.out_activ(z_micro_var_norm)

class DriftMacroNetGraph(nn.Module):
    def __init__(self, config, shape_z, dim_z_macro, dim_z_micro_activ):
        super(DriftMacroNetGraph, self).__init__()
        self.shape_z = shape_z
        self.dim_z_macro = dim_z_macro
        self.dim_z_micro_activ = dim_z_micro_activ
        self.nfe = 0

        self.w = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.r = 2
        dim_z_grid = int(np.prod(self.shape_z[1:]))

        _adj = torch.zeros((*self.shape_z[1:], *self.shape_z[1:])).to(torch.bool)
        for i in range(self.shape_z[1]):
            for j in range(self.shape_z[2]):
                for ir in range(i - self.r, i + self.r + 1):
                    for jr in range(j - self.r, j + self.r + 1):
                        _adj[i, j, ir % self.shape_z[1], jr % self.shape_z[2]] = True
        _adj = _adj.flatten(0, 1).flatten(1, 2)

        self.adj = _adj.nonzero(as_tuple=True)[1].reshape(dim_z_grid, (2*self.r + 1)**2)

        self.macro_net = nn.Sequential(nn.Linear(self.shape_z[0]*(2*self.r + 1)**2 + self.dim_z_micro_activ, 128),
                                       nn.LeakyReLU(),
                                       nn.Linear(128, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 128),
                                       nn.LeakyReLU(),
                                       nn.Linear(128, self.shape_z[0]))
        self.macro_vmap = torch.vmap(self.macro_drift, in_dims=(2, 1, 1, 1), out_dims=2)
        
        for layer in self.macro_net:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def macro_drift(self,
                    z_macro_adj: Float[Tensor, "n_batch n_chan dim_z_macro_adj"],
                    z_micro_activ: Float[Tensor, "n_batch dim_z_micro_activ"],
                    ij: Float[Tensor, "n_batch 2"],
                    t: Float[Tensor, "n_batch 1"]
    ) -> Float[Tensor, "n_batch n_chan"]:
        dz = self.macro_net(torch.cat([z_macro_adj.flatten(1),
                                       #torch.cos(2*torch.pi*ij[:, 0]/self.shape_z[1]).unsqueeze(1),
                                       #torch.sin(2*torch.pi*ij[:, 0]/self.shape_z[1]).unsqueeze(1),
                                       #torch.cos(2*torch.pi*ij[:, 1]/self.shape_z[2]).unsqueeze(1),
                                       #torch.sin(2*torch.pi*ij[:, 1]/self.shape_z[2]).unsqueeze(1),
                                       z_micro_activ], dim=-1))
        return dz
    
    @jaxtyped(typechecker=beartype)
    def forward(self,
                 mu: Float[Tensor, "n_batch dim_mu"],
                 t: Float[Tensor, "n_batch 1"],
                 z_macro: Float[Tensor, "n_batch dim_z_macro"],
                 z_micro_activ: Float[Tensor, "n_batch dim_z_micro_activ"],
                 f: Float[Tensor, "n_batch dim_f"]
    ) -> Float[Tensor, "n_batch dim_z_macro"]:
        self.nfe += 1
        
        dim_z_grid = int(np.prod(self.shape_z[1:]))

        z_macro_reshaped = z_macro.reshape(z_macro.shape[0], self.shape_z[0], dim_z_grid)[:, :, self.adj]
        z_micro_activ_reshaped = z_micro_activ.unsqueeze(1).expand(-1, dim_z_grid, -1)

        ij = torch.stack(torch.meshgrid(torch.arange(self.shape_z[1]), torch.arange(self.shape_z[2])), dim=-1).to(z_macro.device)
        ij_reshaped = ij.flatten(0, 1).unsqueeze(0).expand(z_macro.shape[0], -1, -1)

        t_reshaped = t.unsqueeze(1).expand(-1, dim_z_grid, -1)

        dzdt_macro = self.macro_vmap(z_macro_reshaped, z_micro_activ_reshaped, ij_reshaped, t_reshaped).flatten(1)

        return dzdt_macro

class DriftMicroNet(nn.Module):
    def __init__(self, config, shape_z, dim_z_micro, dim_z_macro_activ):
        super(DriftMicroNet, self).__init__()
        self.shape_z = shape_z
        self.dim_z_micro = dim_z_micro
        
        self.micro_net = nn.Sequential(nn.Linear(dim_z_micro + dim_z_macro_activ + 1, 128),
                                       nn.LeakyReLU(),
                                       nn.Linear(128, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 128),
                                       nn.LeakyReLU(),
                                       nn.Linear(128, dim_z_micro),
        )

        for layer in self.micro_net:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                t: Float[Tensor, "n_batch 1"],
                z_micro: Float[Tensor, "n_batch dim_z_micro"],
                z_macro_activ: Float[Tensor, "n_batch dim_z_macro_activ"],
                f: Float[Tensor, "n_batch dim_f"]
    ) -> Float[Tensor, "n_batch dim_z_micro"]:
        z_interleave = torch.stack([z_macro_activ, z_micro], dim=-1).flatten(1)
        dzdt_micro = self.micro_net(torch.cat([t, z_interleave], dim=1))
        
        return dzdt_micro.flatten(1)

class DriftMacroActivNet(nn.Module):
    def __init__(self, config, dim_z_macro):
        super(DriftMacroActivNet, self).__init__()
        dim_z_micro = config.dim_z - dim_z_macro
        self.macro_activ_net = nn.Linear(dim_z_macro, dim_z_micro)

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                t: Float[Tensor, "n_batch 1"],
                z_macro: Float[Tensor, "n_batch dim_z_macro"],
                f: Float[Tensor, "n_batch dim_f"]
    ) -> Float[Tensor, "n_batch dim_z_macro_activ"]:
        return self.macro_activ_net(z_macro)

class DriftMicroActivNet(nn.Module):
    def __init__(self, config, dim_z_micro):
        super(DriftMicroActivNet, self).__init__()
        self.micro_activ_net = nn.Identity()

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                t: Float[Tensor, "n_batch 1"],
                z_micro: Float[Tensor, "n_batch dim_z_micro"],
                f: Float[Tensor, "n_batch dim_f"],
    ) -> Float[Tensor, "n_batch dim_z_micro_activ"]:
        return self.micro_activ_net(z_micro)

class DispMacroNet(nn.Module):
    def __init__(self, config, dim_z_macro):
        super(DispMacroNet, self).__init__()

        self.out_activ = nn.Softplus()
        self.fixed_disp = nn.Parameter(torch.zeros((1, dim_z_macro)))

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                t: Float[Tensor, "n_batch 1"]
    ) -> Float[Tensor, "n_batch dim_z_macro"]:
        disp_norm = self.fixed_disp.expand(mu.shape[0], *self.fixed_disp.shape[1:])
        return self.out_activ(disp_norm)

class DispMicroNet(nn.Module):
    def __init__(self, config, dim_z_micro):
        super(DispMicroNet, self).__init__()

        self.out_activ = nn.Softplus()
        self.fixed_disp = nn.Parameter(torch.zeros((1, dim_z_micro)))

    @jaxtyped(typechecker=beartype)
    def forward(self,
                mu: Float[Tensor, "n_batch dim_mu"],
                t: Float[Tensor, "n_batch 1"]
    ) -> Float[Tensor, "n_batch dim_z_micro"]:
        disp_norm = self.fixed_disp.expand(mu.shape[0], *self.fixed_disp.shape[1:])
        return self.out_activ(disp_norm)

class KernelNet(nn.Module):
    def __init__(self, config):
        super(KernelNet, self).__init__()
        '''
        self.net = nn.Sequential(nn.Linear(1, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 1))
        '''
        self.net = nn.Identity()
    
    def forward(self, t: Tensor) -> Tensor:
        return self.net(t)

def augment_latent_sde(old_config: CaseConfig, config: CaseConfig, device: torch.device = torch.device("cuda:0")) -> visde.LatentSDE:

    if old_config.dim_z_micro >= 0:
        old_dummy_model = create_latent_sde(old_config, device)
        old_version = get_version_str(old_config)

        ckpt_dir = os.path.join(CURR_DIR, "logs_visde", old_version, "checkpoints")
        for file in os.listdir(ckpt_dir):
            if file.endswith(".ckpt"):
                ckpt_file = file
        
        old_model = visde.LatentSDE.load_from_checkpoint(os.path.join(ckpt_dir, ckpt_file),
                                                        config=old_dummy_model.config,
                                                        encoder=old_dummy_model.encoder,
                                                        decoder=old_dummy_model.decoder,
                                                        drift=old_dummy_model.drift,
                                                        dispersion=old_dummy_model.dispersion,
                                                        loglikelihood=old_dummy_model.loglikelihood,
                                                        latentvar=old_dummy_model.latentvar).to(device)
        
        # copy all parameters with augmented dim_z_micro
        dim_z = config.dim_z_macro + config.dim_z_micro

        new_model = create_latent_sde(config, device)

        with torch.no_grad():
            state_dict = old_model.state_dict()

            # encoder var
            print(state_dict["encoder.micro_var_net.fixed_var"].shape)
            new_var = -4*torch.ones((1, config.dim_z_micro))
            new_var[:, :config.dim_z_micro-1] = state_dict["encoder.micro_var_net.fixed_var"]
            state_dict["encoder.micro_var_net.fixed_var"] = new_var

            # drift
            new_weight = torch.zeros((128, 25 + config.dim_z_micro)) # 29 if ij included
            new_weight[:, :25 + config.dim_z_micro - 1] = state_dict["drift.macro_drift_net.macro_net.0.weight"]
            torch.nn.init.xavier_normal_(new_weight[:, 25 + config.dim_z_micro - 1:])
            state_dict["drift.macro_drift_net.macro_net.0.weight"] = new_weight

            new_weight = torch.zeros((128, 1 + 2*config.dim_z_micro))
            new_weight[:, :(1 + 2*(config.dim_z_micro-1))] = state_dict["drift.micro_drift_net.micro_net.0.weight"]
            torch.nn.init.xavier_normal_(new_weight[:, (1 + 2*(config.dim_z_micro-1)):])
            state_dict["drift.micro_drift_net.micro_net.0.weight"] = new_weight

            new_weight = torch.zeros((config.dim_z_micro, 128))
            new_weight[:config.dim_z_micro-1] = state_dict["drift.micro_drift_net.micro_net.6.weight"]
            torch.nn.init.xavier_normal_(new_weight[config.dim_z_micro-1:])
            state_dict["drift.micro_drift_net.micro_net.6.weight"] = new_weight

            new_bias = torch.zeros((config.dim_z_micro,))
            new_bias[:config.dim_z_micro-1] = state_dict["drift.micro_drift_net.micro_net.6.bias"]
            torch.nn.init.zeros_(new_bias[config.dim_z_micro-1:])
            state_dict["drift.micro_drift_net.micro_net.6.bias"] = new_bias

            new_weight = torch.zeros((config.dim_z_micro, config.dim_z_macro))
            new_weight[:config.dim_z_micro-1] = state_dict["drift.macro_activ_net.macro_activ_net.weight"]
            torch.nn.init.xavier_normal_(new_weight[config.dim_z_micro-1:])
            state_dict["drift.macro_activ_net.macro_activ_net.weight"] = new_weight

            new_bias = torch.zeros((config.dim_z_micro,))
            new_bias[:config.dim_z_micro-1] = state_dict["drift.macro_activ_net.macro_activ_net.bias"]
            torch.nn.init.zeros_(new_bias[config.dim_z_micro-1:])
            state_dict["drift.macro_activ_net.macro_activ_net.bias"] = new_bias

            # dispersion
            new_disp = torch.ones((1, config.dim_z_micro))
            new_disp[:, :config.dim_z_micro-1] = state_dict["dispersion.micro_net.fixed_disp"]
            torch.nn.init.xavier_normal_(new_disp[:, config.dim_z_micro-1:])
            state_dict["dispersion.micro_net.fixed_disp"] = new_disp

            # latentvar
            new_var = -4*torch.ones((1, config.dim_z_micro))
            new_var[:, :config.dim_z_micro-1] = state_dict["latentvar.encoder.micro_var_net.fixed_var"]
            state_dict["latentvar.encoder.micro_var_net.fixed_var"] = new_var

            if old_config.dim_z_micro >= 1:
                new_weight = torch.zeros((config.dim_z_micro, 4096))
                new_weight[:config.dim_z_micro-1] = state_dict["encoder.micro_mean_net.net.8.weight"]
                torch.nn.init.xavier_normal_(new_weight[config.dim_z_micro-1:])
                state_dict["encoder.micro_mean_net.net.8.weight"] = new_weight

                new_bias = torch.zeros((config.dim_z_micro,))
                new_bias[:config.dim_z_micro-1] = state_dict["encoder.micro_mean_net.net.8.bias"]
                torch.nn.init.zeros_(new_bias[config.dim_z_micro-1:])
                state_dict["encoder.micro_mean_net.net.8.bias"] = new_bias

                new_weight = torch.zeros((4096, config.dim_z_micro))
                new_weight[:, :config.dim_z_micro-1] = state_dict["decoder.micro_mean_net.net.0.weight"]
                torch.nn.init.xavier_normal_(new_weight[:, config.dim_z_micro-1:])
                state_dict["decoder.micro_mean_net.net.0.weight"] = new_weight

                new_weight = torch.zeros((config.dim_z_micro, 4096))
                new_weight[:config.dim_z_micro-1] = state_dict["latentvar.encoder.micro_mean_net.net.8.weight"]
                torch.nn.init.xavier_normal_(new_weight[config.dim_z_micro-1:])
                state_dict["latentvar.encoder.micro_mean_net.net.8.weight"] = new_weight

                new_bias = torch.zeros((config.dim_z_micro,))
                new_bias[:config.dim_z_micro-1] = state_dict["latentvar.encoder.micro_mean_net.net.8.bias"]
                torch.nn.init.zeros_(new_bias[config.dim_z_micro-1:])
                state_dict["latentvar.encoder.micro_mean_net.net.8.bias"] = new_bias

            new_model.load_state_dict(state_dict, strict=False)
        
        return new_model

    else:
        return create_latent_sde(config, device)

def create_latent_sde(case_config: CaseConfig, device: torch.device = torch.device("cuda:0")
) -> visde.LatentSDE:
    data_path = full_data_path(case_config.data_file)
    dataset = visde.MultiTrajHDF5(data_path, "val", case_config.n_win)
    
    dim_z_macro = case_config.dim_z_macro
    dim_z_micro = case_config.dim_z_micro
    n_batch = case_config.n_batch
    n_sigma = case_config.n_sigma
    n_win = case_config.n_win

    shape_x = dataset.shape_x
    dim_x = int(np.prod(shape_x))
    dim_z = dim_z_macro + dim_z_micro

    dim_mu = dataset.dim_mu
    dim_f = dataset.dim_f
    dt = dataset.dt

    n_chan = shape_x[0]
    sigma = int(np.sqrt(dim_x//dim_z_macro) + 0.5)
    shape_z = (shape_x[0], *[grid_ax // sigma for grid_ax in shape_x[1:]])

    kernel_range_x = torch.arange(-n_sigma*sigma, n_sigma*sigma + 1).view(1, 1, -1, 1).tile((n_chan, 1, 1, 1))
    kernel_range_y = torch.arange(-n_sigma*sigma, n_sigma*sigma + 1).view(1, 1, 1, -1).tile((n_chan, 1, 1, 1))
    shared_kernel = nn.Parameter(torch.exp(-(kernel_range_x**2 + kernel_range_y**2)/(2*sigma**2))/
                                np.sqrt(n_chan*(2*n_sigma*sigma + 1)*(2*n_sigma*sigma + 1)))
    shared_bias = nn.Parameter(torch.zeros(n_chan))

    # encoder
    vaeconfig = visde.VarAutoencoderConfig(dim_mu=dim_mu, dim_x=dim_x, dim_z=dim_z, n_win=n_win, shape_x=shape_x)
    encoder_mean_macro_net = EncodeMeanMacroNet(vaeconfig, dim_z_macro, shared_kernel, shared_bias, n_sigma)
    encoder_mean_micro_net = EncodeMeanMicroNet(vaeconfig, dim_z_micro)
    encoder_var_macro_net = EncodeVarMacroNet(vaeconfig, dim_z_macro)
    encoder_var_micro_net = EncodeVarMicroNet(vaeconfig, dim_z_micro)
    encoder_smooth_net = EncodeSmoothNet(vaeconfig, dim_z_macro, shared_kernel, shared_bias, n_sigma)
    encoder = multiscale.MultiscaleVarEncoderNoPrior(vaeconfig, dim_z_macro, dim_z_micro, encoder_mean_macro_net,
                                                     encoder_var_macro_net, encoder_mean_micro_net,
                                                     encoder_var_micro_net, encoder_smooth_net)

    # decoder
    decoder_mean_macro_net = DecodeMeanMacroNet(vaeconfig, shape_z, dim_z_macro, shared_kernel, shared_bias, n_sigma)
    decoder_mean_micro_net = DecodeMeanMicroNet(vaeconfig, dim_z_micro)
    decoder_var_macro_net = DecodeVarMacroNet(vaeconfig)
    decoder_var_micro_net = DecodeVarMicroNet(vaeconfig)
    decoder = multiscale.MultiscaleVarDecoderNoPrior(vaeconfig, dim_z_macro, dim_z_micro,
                                                     decoder_mean_macro_net, decoder_var_macro_net,
                                                     decoder_mean_micro_net, decoder_var_micro_net)

    # drift
    config = visde.LatentDriftConfig(dim_mu=dim_mu, dim_z=dim_z, dim_f=dim_f)
    drift_macro_net = DriftMacroNetGraph(config, shape_z, dim_z_macro, dim_z_micro)
    drift_micro_net = DriftMicroNet(config, shape_z, dim_z_micro, dim_z_micro)
    drift_macro_activ_net = DriftMacroActivNet(config, dim_z_macro)
    drift_micro_activ_net = DriftMicroActivNet(config, dim_z_micro)
    drift = multiscale.MultiscaleLatentDriftNoPrior(config, dim_z_macro, dim_z_micro,
                                                    drift_macro_net, drift_micro_net,
                                                    drift_macro_activ_net, drift_micro_activ_net)

    # dispersion
    config = visde.LatentDispersionConfig(dim_mu=dim_mu, dim_z=dim_z)
    disp_macro_net = DispMacroNet(config, dim_z_macro)
    disp_micro_net = DispMicroNet(config, dim_z_micro)
    dispersion = multiscale.MultiscaleLatentDispersionNoPrior(config, dim_z_macro, dim_z_micro,
                                                              disp_macro_net, disp_micro_net)

    # likelihood
    loglikelihood = visde.LogLikeGaussian()

    # latent variational distribution
    config = visde.LatentVarConfig(dim_mu=dim_mu, dim_z=dim_z)
    kernel_net = KernelNet(config)
    kernel = visde.DeepGaussianKernel(kernel_net, n_batch, dt)
    latentvar = visde.AmortizedLatentVarGP(config, kernel, encoder)
    #latentvar = visde.ParamFreeLatentVarGP(config, encoder)

    config = visde.LatentSDEConfig(n_totaldata=dataset.total_data,
                                   n_samples=1,
                                   n_tquad=0,
                                   n_warmup=0,
                                   n_transition=1800,
                                   lr=case_config.lr,
                                   lr_sched_freq=case_config.lr_sched_freq)
    model = visde.LatentSDE(config=config,
                            encoder=encoder,
                            decoder=decoder,
                            drift=drift,
                            dispersion=dispersion,
                            loglikelihood=loglikelihood,
                            latentvar=latentvar).to(device)
    
    return model

