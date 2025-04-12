import torch
from torch import Tensor, nn
from jaxtyping import Float
import matplotlib.pyplot as plt

import pickle as pkl
import numpy as np
import os
import pathlib

import visde
# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def softplus(x: Tensor) -> Tensor:
    return torch.log(1 + torch.exp(x))

class EncodeMeanNet(nn.Module):
    def __init__(self, config, dim_z_macro, dim_z_micro, shared_kernel, shared_bias, n_sigma):
        super(EncodeMeanNet, self).__init__()
        n_win = 1#config.n_win
        dim_x = config.dim_x

        self.dim_x = dim_x
        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro
        self.sigma = self.dim_x//self.dim_z_macro

        self.smooth_net = nn.Sequential(nn.Unflatten(1, (1, self.dim_x)),
                                        nn.Conv1d(n_win, 1, kernel_size=2*n_sigma*self.sigma + 1, stride=1, padding_mode="circular", padding=n_sigma*self.sigma),
                                        nn.Flatten())

        self.macro_net = nn.Sequential(nn.Unflatten(1, (1, self.dim_x)),
                                       nn.Conv1d(n_win, 1, kernel_size=2*n_sigma*self.sigma + 1, stride=self.sigma, padding_mode="circular", padding=n_sigma*self.sigma),
                                       nn.Flatten())

        self.smooth_net[1].weight = shared_kernel
        self.smooth_net[1].bias = shared_bias
        self.macro_net[1].weight = shared_kernel
        self.macro_net[1].bias = shared_bias

        self.micro_net = nn.Sequential(nn.Unflatten(1, (1, self.dim_x)),
                                       nn.Conv1d(1, 4, kernel_size=25, stride=2, padding=12, padding_mode="circular"),
                                       nn.ReLU(),
                                       nn.Conv1d(4, 16, kernel_size=25, stride=2, padding=12, padding_mode="circular"),
                                       nn.ReLU(),
                                       nn.Conv1d(16, 64, kernel_size=25, stride=2, padding=12, padding_mode="circular"),
                                       nn.ReLU(),
                                       nn.Flatten(),
                                       nn.Linear(64*125, self.dim_z_micro)
                                       )

        for layer in self.micro_net:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        n_batch = x_win.shape[0]
        x = x_win.flatten(1)

        z_macro = self.macro_net(x)
        x_smooth = self.smooth_net(x)
        x_delta = x - x_smooth

        z_micro = self.micro_net(x_delta)

        '''
        plt.figure()
        dim_x = x_win.shape[-1]
        sigma = dim_x//z_macro.shape[-1]
        x_domain = np.linspace(0, 1, dim_x + 1)[:dim_x]
        plt.plot(x_domain, x_win[0, 0, :].cpu().detach().numpy(), color='blue', label='True')
        plt.plot(x_domain, self.smooth_net(x_win.flatten(1))[0, :].cpu().detach().numpy(), color='red', label='Smooth')
        plt.plot(x_domain[::sigma], z_macro[0, :].cpu().detach().numpy(), color='green', label='Macro')
        plt.ylim(-0.5, 1.5)
        plt.legend()
        plt.show()
        plt.savefig('blur.png')
        plt.close()
        '''

        return torch.cat([z_macro, z_micro], dim=-1)

class EncodeVarNet(nn.Module):
    def __init__(self, config, dim_z_macro, dim_z_micro):
        super(EncodeVarNet, self).__init__()
        dim_z = config.dim_z

        self.out_activ = nn.Softplus()
        self.fixed_var = nn.Parameter(-4*torch.ones((1, dim_z)))

    def forward(self, mu: Tensor, x_win: Tensor) -> Tensor:
        z_var_norm = self.fixed_var.expand(x_win.shape[0], *self.fixed_var.shape[1:])
        return self.out_activ(z_var_norm)

class DecodeMeanNet(nn.Module):
    def __init__(self, config, dim_z_macro, dim_z_micro, _macro_var, _micro_var, shared_kernel, shared_bias, n_sigma):
        super(DecodeMeanNet, self).__init__()
        self.dim_x = config.dim_x
        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro
        self.sigma = self.dim_x//self.dim_z_macro

        self.macro_net = nn.Sequential(nn.Unflatten(1, (1, self.dim_z_macro)),
                                       nn.ConvTranspose1d(1, 1, kernel_size=2*n_sigma*self.sigma + 1, stride=self.sigma, padding=n_sigma*self.sigma, output_padding=self.sigma-1),
                                       nn.Flatten())
    
        self.macro_net[1].weight = shared_kernel
        self.macro_net[1].bias = shared_bias

        self.micro_net = nn.Sequential(nn.Linear(self.dim_z_micro, 64*125),
                                       nn.ReLU(),
                                       nn.Unflatten(1, (64, 125)),
                                       nn.ConvTranspose1d(64, 16, kernel_size=25, stride=2, padding=12, output_padding=1),
                                       nn.ReLU(),
                                       nn.ConvTranspose1d(16, 4, kernel_size=25, stride=2, padding=12, output_padding=1),
                                       nn.ReLU(),
                                       nn.ConvTranspose1d(4, 1, kernel_size=25, stride=2, padding=12, output_padding=1),
                                       nn.Flatten()
                                       )
        
        self.out_activ = nn.Softplus()
        self._macro_var = _macro_var
        self._micro_var = _micro_var

        for layer in self.micro_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def decode_macro(self, z_macro: Tensor) -> Tensor:
        return self.macro_net(z_macro)

    def decode_micro(self, z_micro: Tensor) -> Tensor:
        n_batch = z_micro.shape[0]

        if self.dim_z_micro == 0:
            x_micro = torch.zeros(n_batch, self.dim_x, device=z_micro.device)
        else:
            x_micro_unnorm = self.micro_net(z_micro)

            batch_macro_var = self.macro_var.expand(n_batch, *self.macro_var.shape[1:])
            batch_micro_var = self.micro_var.expand(n_batch, *self.micro_var.shape[1:])
            joint_var = (batch_macro_var.pow(-1) + batch_micro_var.pow(-1)).pow(-1)

            x_micro = x_micro_unnorm.mul(joint_var).div(batch_micro_var)
        
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

        x_macro = self.decode_macro(z_macro)
        x_micro = self.decode_micro(z_micro)

        x = x_macro + x_micro

        '''
        plt.figure()
        sigma = self.dim_x//self.dim_z_macro
        x_domain = np.linspace(0, 1, self.dim_x + 1)[:self.dim_x]
        plt.plot(x_domain[::sigma], z_macro[0, :].cpu().detach().numpy(), color='blue')
        plt.plot(x_domain, x_macro[0, :].cpu().detach().numpy(), color='red')
        plt.plot(x_domain, x[0, :].cpu().detach().numpy(), color='green')
        plt.ylim(-1.5, 1.5)
        plt.legend(['Coarse', 'Macro', 'Reconstructed'])
        plt.show()
        plt.savefig('deblur.png')
        plt.close()
        '''

        return x

class DecodeVarNet(nn.Module):
    def __init__(self, config, _macro_var, _micro_var):
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
    def __init__(self, config, dim_x, dim_z_macro, dim_z_micro):
        super(DriftNet, self).__init__()
        self.dim_z = config.dim_z
        self.dim_z_macro = dim_z_macro
        self.dim_z_micro = dim_z_micro
        self.dim_x = dim_x
        self.radius = 2

        self.fcnet_macro = nn.Sequential(nn.Linear(2*self.radius + 1 + self.dim_z_micro, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 1))

        self.fcnet_micro = nn.Sequential(nn.Linear(self.dim_z_macro + self.dim_z_micro + 2, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, self.dim_z_micro))

        self.macro_vmap = torch.vmap(self.fcnet_macro, in_dims=1, out_dims=1)

        self.micro_activ = nn.Identity() #nn.Sequential(nn.Tanh(),
        #                                 nn.Linear(self.dim_z_micro, self.dim_z_micro))

        self.macro_activ = nn.Identity() #nn.Sequential(nn.Linear(self.dim_z_macro, self.dim_z_micro),
        #                                 nn.Tanh())
        
        for layer in self.fcnet_macro:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.fcnet_micro:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,
                 mu: Float[Tensor, "n_batch dim_mu"],
                 t: Float[Tensor, "n_batch 1"],
                 z: Float[Tensor, "n_batch dim_z"],
                 f: Float[Tensor, "n_batch dim_f"]
    ) -> Float[Tensor, "n_batch dim_z"]:
        n_batch = z.shape[0]
        z_macro = z[:, 0:self.dim_z_macro]

        for r in range(-self.radius, self.radius + 1):
            ir = (np.arange(self.dim_z_macro) + r) % self.dim_z_macro
            if r == -self.radius:
                z_macro_stack = z_macro[:, ir].unsqueeze(-1)
            else:
                z_macro_stack = torch.cat([z_macro_stack, z_macro[:, ir].unsqueeze(-1)], dim=-1)

        z_micro = z[:, self.dim_z_macro:]
        z_micro_stack = self.micro_activ(z_micro).unsqueeze(1).expand(-1, self.dim_z_macro, -1)

        #i_range = torch.linspace(0, 1, self.dim_z_macro, device=z.device).unsqueeze(0).expand(n_batch, -1).unsqueeze(-1)

        dzdt_macro = self.macro_vmap(torch.cat([z_macro_stack, z_micro_stack], dim=-1)).flatten(1)
        dzdt_micro = self.fcnet_micro(torch.cat([z_micro, self.macro_activ(z_macro), torch.sin(t), torch.cos(t)], dim=-1))

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
                      n_sigma: int,
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
    dt = data["train_t"][0,1] - data["train_t"][0,0]

    sigma = dim_x//dim_z_macro
    kernel_range = torch.arange(-n_sigma*sigma, n_sigma*sigma + 1).view(1, 1, -1)
    shared_kernel = nn.Parameter(torch.exp(-(kernel_range**2)/(2*sigma**2))/np.sqrt(2*n_sigma*sigma + 1))
    shared_bias = nn.Parameter(torch.zeros(1))

    # encoder
    vaeconfig = visde.VarAutoencoderConfig(dim_mu=dim_mu, dim_x=dim_x, dim_z=dim_z, n_win=n_win, shape_x=shape_x)
    encode_mean_net = EncodeMeanNet(vaeconfig, dim_z_macro, dim_z_micro, shared_kernel, shared_bias, n_sigma)
    encode_var_net = EncodeVarNet(vaeconfig, dim_z_macro, dim_z_micro)
    encoder = visde.VarEncoderNoPrior(vaeconfig, encode_mean_net, encode_var_net)

    # decoder
    _macro_var = nn.Parameter(-4*torch.ones((1, *shape_x)))
    _micro_var = nn.Parameter(-4*torch.ones((1, *shape_x)))
    decode_mean_net = DecodeMeanNet(vaeconfig, dim_z_macro, dim_z_micro, _macro_var, _micro_var, shared_kernel, shared_bias, n_sigma)
    decode_var_net = DecodeVarNet(vaeconfig, _macro_var, _micro_var)
    decoder = visde.VarDecoderNoPrior(vaeconfig, decode_mean_net, decode_var_net)

    # drift
    config = visde.LatentDriftConfig(dim_mu=dim_mu, dim_z=dim_z, dim_f=dim_f)
    driftnet = DriftNet(config, dim_x, dim_z_macro, dim_z_micro)
    drift = visde.LatentDriftNoPrior(config, driftnet)

    # dispersion
    config = visde.LatentDispersionConfig(dim_mu=dim_mu, dim_z=dim_z)
    dispnet = DispNet(config, dim_z_macro, dim_z_micro)
    dispersion = visde.LatentDispersionNoPrior(config, dispnet)

    # likelihood
    loglikelihood = visde.LogLikeGaussian()

    # latent variational distribution
    config = visde.LatentVarConfig(dim_mu=dim_mu, dim_z=dim_z)
    kernel_net = KernelNet(config)
    kernel = visde.DeepGaussianKernel(kernel_net, n_batch, dt)
    latentvar = visde.AmortizedLatentVarGP(config, kernel, encoder)

    config = visde.LatentSDEConfig(n_totaldata=torch.numel(data["train_t"]),
                                   n_samples=1,
                                   n_tquad=10,
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

