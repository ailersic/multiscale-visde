import torch
#import torchvision
from torch import Tensor, nn
from torch.optim import lr_scheduler
from jaxtyping import Float, jaxtyped
from beartype import beartype
from kolsol.torch.solver import KolSol
from matplotlib import pyplot as plt

import numpy as np
import os
import pathlib
import math

import visde
import multiscale
# ruff: noqa: F821, F722

from experiments_refac.kdv_1d.utils import CaseConfig, get_version_str
from datasets.kdv_1d.load_data import load_data

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def softplus(x: Tensor) -> Tensor:
    return torch.log(1 + torch.exp(x))

class EncodeMeanMacroNet(nn.Module):
    def __init__(self, config, dim_z_macro, shared_kernel, shared_bias, n_sigma):
        super(EncodeMeanMacroNet, self).__init__()
        sigma = config.dim_x//dim_z_macro

        self.net = nn.Sequential(nn.Flatten(1, 2),
                                 nn.Conv1d(config.n_win, 1, kernel_size=2*n_sigma*sigma + 1, stride=sigma,
                                           padding_mode="circular", padding=n_sigma*sigma),
                                 nn.Flatten())

        self.net[1].weight = shared_kernel
        self.net[1].bias = shared_bias

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        out = self.net(x_win)
        return out

class EncodeSmoothNet(nn.Module):
    def __init__(self, config, dim_z_macro, shared_kernel, shared_bias, n_sigma):
        super(EncodeSmoothNet, self).__init__()
        sigma = config.dim_x//dim_z_macro

        self.net = nn.Sequential(nn.Flatten(1, 2),
                                 nn.Conv1d(config.n_win, 1, kernel_size=2*n_sigma*sigma + 1, stride=1,
                                           padding_mode="circular", padding=n_sigma*sigma),
                                 nn.Unflatten(1, (1, 1)))

        self.net[1].weight = shared_kernel
        self.net[1].bias = shared_bias

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        return self.net(x_win)

class EncodeMeanMicroNet(nn.Module):
    def __init__(self, config, dim_z_micro):
        super(EncodeMeanMicroNet, self).__init__()

        micro_padding = 12
        micro_kernel = 2*micro_padding + 1
        self.dim_z_micro = dim_z_micro
        dim_hidden = 8 * config.dim_x

        if dim_z_micro > 0:
            self.net = nn.Sequential(nn.Flatten(1, 2),
                                     nn.Conv1d(config.n_win, 4, kernel_size=micro_kernel, stride=2, padding=micro_padding, padding_mode="circular"),
                                     nn.LeakyReLU(),
                                     nn.Conv1d(4, 16, kernel_size=micro_kernel, stride=2, padding=micro_padding, padding_mode="circular"),
                                     nn.LeakyReLU(),
                                     nn.Conv1d(16, 64, kernel_size=micro_kernel, stride=2, padding=micro_padding, padding_mode="circular"),
                                     nn.LeakyReLU(),
                                     nn.Flatten(),
                                     nn.Linear(dim_hidden, dim_z_micro)
                                     )

            for layer in self.net:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
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
        self.dim_z_macro = dim_z_macro
        self.sigma = config.dim_x//dim_z_macro
        self.n_sigma = n_sigma
        self.shape_z = shape_z

        self.net = nn.Sequential(nn.Unflatten(1, self.shape_z),
                                 nn.CircularPad1d(n_sigma),
                                 nn.ConvTranspose1d(1, 1, kernel_size=2*n_sigma*self.sigma + 1, stride=self.sigma,
                                                    padding=n_sigma*self.sigma, output_padding=self.sigma-1)
                                )

        self.net[2].weight = shared_kernel
        self.net[2].bias = shared_bias

    def forward(self, mu: Tensor, z_macro: Tensor) -> Tensor:
        return self.net(z_macro)[:, :, self.n_sigma*self.sigma:-self.n_sigma*self.sigma]

class DecodeMeanMicroNet(nn.Module):
    def __init__(self, config, dim_z_micro):
        super(DecodeMeanMicroNet, self).__init__()

        self.shape_x = config.shape_x
        micro_padding = 4
        micro_kernel = 2*micro_padding + 1
        dim_hidden = 8 * config.dim_x

        self.dim_z_micro = dim_z_micro

        if dim_z_micro > 0:
            self.net = nn.Sequential(nn.Linear(dim_z_micro, dim_hidden),
                                     nn.LeakyReLU(),
                                     nn.Unflatten(1, (64, 125)),
                                     nn.ConvTranspose1d(64, 16, kernel_size=micro_kernel, stride=2, padding=micro_padding, output_padding=1),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose1d(16, 4, kernel_size=micro_kernel, stride=2, padding=micro_padding, output_padding=1),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose1d(4, 1, kernel_size=micro_kernel, stride=2, padding=micro_padding, output_padding=1)
                                     )

            for layer in self.net:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose1d):
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

class DriftMacroNet(nn.Module):
    def __init__(self, config, shape_z, dim_z_macro, dim_z_micro_activ):
        super(DriftMacroNet, self).__init__()
        self.dim_z = config.dim_z
        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro_activ
        self.radius = 2
        self.nfe = 0

        self.fcnet_macro = nn.Sequential(nn.Linear(2*self.radius + 1 + self.dim_z_micro, 128),
                                        nn.LeakyReLU(),
                                        nn.Linear(128, 128),
                                        nn.LeakyReLU(),
                                        nn.Linear(128, 128),
                                        nn.LeakyReLU(),
                                        nn.Linear(128, 1))

        self.macro_vmap = torch.vmap(self.fcnet_macro, in_dims=1, out_dims=1)
        
        for layer in self.fcnet_macro:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    @jaxtyped(typechecker=beartype)
    def forward(self,
                 mu: Float[Tensor, "n_batch dim_mu"],
                 t: Float[Tensor, "n_batch 1"],
                 z_macro: Float[Tensor, "n_batch dim_z_macro"],
                 z_micro_activ: Float[Tensor, "n_batch dim_z_micro_activ"],
                 f: Float[Tensor, "n_batch dim_f"]
    ) -> Float[Tensor, "n_batch dim_z_macro"]:
        self.nfe += 1
        
        for r in range(-self.radius, self.radius + 1):
            ir = (np.arange(self.dim_z_macro) + r) % self.dim_z_macro
            if r == -self.radius:
                z_macro_stack = z_macro[:, ir].unsqueeze(-1)
            else:
                z_macro_stack = torch.cat([z_macro_stack, z_macro[:, ir].unsqueeze(-1)], dim=-1)
        
        z_micro_stack = z_micro_activ.unsqueeze(1).expand(-1, self.dim_z_macro, -1)
        #t_stack = t.unsqueeze(1).expand(-1, self.dim_z_macro, -1)
        dzdt_macro = self.macro_vmap(torch.cat([z_macro_stack,
                                                #torch.cos(2*torch.pi*t_stack),
                                                #torch.sin(2*torch.pi*t_stack),
                                                z_micro_stack], dim=-1)).flatten(1)

        return dzdt_macro

class DriftMicroNet(nn.Module):
    def __init__(self, config, shape_z, dim_z_micro, dim_z_macro_activ):
        super(DriftMicroNet, self).__init__()
        self.shape_z = shape_z
        self.dim_z_micro = dim_z_micro
        
        self.fcnet_micro = nn.Sequential(nn.Linear(dim_z_macro_activ + self.dim_z_micro + 1, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, self.dim_z_micro))

        for layer in self.fcnet_micro:
            if isinstance(layer, nn.Linear):
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
        dzdt_micro = self.fcnet_micro(torch.cat([t, z_macro_activ, z_micro], dim=1)) #torch.cos(2*torch.pi*t), torch.sin(2*torch.pi*t),

        return dzdt_micro

class DriftMacroActivNet(nn.Module):
    def __init__(self, config, dim_z_macro):
        super(DriftMacroActivNet, self).__init__()
        self.macro_activ_net = nn.Identity()

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

        self.net = nn.Sequential(nn.Linear(1, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 1))
    
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

            new_var = -4*torch.ones((1, config.dim_z_micro))
            new_var[:, :config.dim_z_micro-1] = state_dict["encoder.micro_var_net.fixed_var"]
            state_dict["encoder.micro_var_net.fixed_var"] = new_var

            new_weight = torch.zeros((128, 5 + config.dim_z_micro))
            new_weight[:, :(5 + config.dim_z_micro - 1)] = state_dict["drift.macro_drift_net.fcnet_macro.0.weight"]
            torch.nn.init.xavier_normal_(new_weight[:, (5 + config.dim_z_micro - 1):])
            state_dict["drift.macro_drift_net.fcnet_macro.0.weight"] = new_weight

            new_weight = torch.zeros((128, dim_z + 1))
            new_weight[:, :dim_z] = state_dict["drift.micro_drift_net.fcnet_micro.0.weight"]
            torch.nn.init.xavier_normal_(new_weight[:, dim_z:])
            state_dict["drift.micro_drift_net.fcnet_micro.0.weight"] = new_weight

            new_weight = torch.zeros((config.dim_z_micro, 128))
            new_weight[:config.dim_z_micro-1, :] = state_dict["drift.micro_drift_net.fcnet_micro.6.weight"]
            torch.nn.init.xavier_normal_(new_weight[config.dim_z_micro-1:, :])
            state_dict["drift.micro_drift_net.fcnet_micro.6.weight"] = new_weight

            new_bias = torch.zeros((config.dim_z_micro,))
            new_bias[:config.dim_z_micro-1] = state_dict["drift.micro_drift_net.fcnet_micro.6.bias"]
            state_dict["drift.micro_drift_net.fcnet_micro.6.bias"] = new_bias

            new_disp = -4*torch.ones((1, config.dim_z_micro))
            new_disp[:, :config.dim_z_micro-1] = state_dict["dispersion.micro_net.fixed_disp"]
            state_dict["dispersion.micro_net.fixed_disp"] = new_disp

            new_var = -4*torch.ones((1, config.dim_z_micro))
            new_var[:, :config.dim_z_micro-1] = state_dict["latentvar.encoder.micro_var_net.fixed_var"]
            state_dict["latentvar.encoder.micro_var_net.fixed_var"] = new_var
            
            if old_config.dim_z_micro >= 1:
                new_weight = torch.zeros((config.dim_z_micro, 8000))
                new_weight[:config.dim_z_micro-1, :] = state_dict["encoder.micro_mean_net.net.8.weight"]
                torch.nn.init.xavier_normal_(new_weight[config.dim_z_micro-1:, :])
                state_dict["encoder.micro_mean_net.net.8.weight"] = new_weight

                new_bias = torch.zeros((config.dim_z_micro,))
                new_bias[:config.dim_z_micro-1] = state_dict["encoder.micro_mean_net.net.8.bias"]
                state_dict["encoder.micro_mean_net.net.8.bias"] = new_bias

                new_weight = torch.zeros((8000, config.dim_z_micro))
                new_weight[:, :config.dim_z_micro-1] = state_dict["decoder.micro_mean_net.net.0.weight"]
                torch.nn.init.xavier_normal_(new_weight[:, config.dim_z_micro-1:])
                state_dict["decoder.micro_mean_net.net.0.weight"] = new_weight

                new_weight = torch.zeros((config.dim_z_micro, 8000))
                new_weight[:config.dim_z_micro-1, :] = state_dict["latentvar.encoder.micro_mean_net.net.8.weight"]
                torch.nn.init.xavier_normal_(new_weight[config.dim_z_micro-1:, :])
                state_dict["latentvar.encoder.micro_mean_net.net.8.weight"] = new_weight

                new_bias = torch.zeros((config.dim_z_micro,))
                new_bias[:config.dim_z_micro-1] = state_dict["latentvar.encoder.micro_mean_net.net.8.bias"]
                state_dict["latentvar.encoder.micro_mean_net.net.8.bias"] = new_bias

            new_model.load_state_dict(state_dict, strict=False)
        
        return new_model

    else:
        return create_latent_sde(config, device)

def create_latent_sde(case_config: CaseConfig, device: torch.device = torch.device("cuda:0")
) -> visde.LatentSDE:
    data = load_data(case_config.data_file)
    
    dim_z_macro = case_config.dim_z_macro
    dim_z_micro = case_config.dim_z_micro
    n_batch = case_config.n_batch
    n_sigma = case_config.n_sigma
    n_win = case_config.n_win

    dim_mu = data["train_mu"].shape[1]
    shape_x = tuple(data["train_x"].shape[2:])
    dim_x = int(np.prod(shape_x))
    dim_z = dim_z_macro + dim_z_micro
    dim_f = data["train_f"].shape[2]
    dt = data["train_t"][0,1] - data["train_t"][0,0]

    n_chan = shape_x[0]
    sigma = dim_x//dim_z_macro
    shape_z = (shape_x[0], *[grid_ax // sigma for grid_ax in shape_x[1:]])

    kernel_range_x = torch.arange(-n_sigma*sigma, n_sigma*sigma + 1).view(1, 1, -1).tile((n_chan, 1, 1))
    unnormalized_kernel = torch.exp(-(kernel_range_x**2)/(2*sigma**2))
    shared_kernel = nn.Parameter(unnormalized_kernel / unnormalized_kernel.sum(dim=-1, keepdim=True))
    shared_bias = nn.Parameter(torch.zeros(n_chan))
    print(f"shared kernel shape: {shared_kernel.shape}, shared bias shape: {shared_bias.shape}")

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
    drift_macro_net = DriftMacroNet(config, shape_z, dim_z_macro, dim_z_micro)
    drift_micro_net = DriftMicroNet(config, shape_z, dim_z_micro, dim_z_macro)
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

    config = visde.LatentSDEConfig(n_totaldata=torch.numel(data["train_t"]),
                                   n_samples=1,
                                   n_tquad=0,
                                   n_warmup=0,
                                   n_transition=1600,
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

