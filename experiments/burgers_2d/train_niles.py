import torch
from torch.utils.data import DataLoader

import pickle as pkl
import os
import pathlib
import shutil

import pytorch_lightning as pl
from pytorch_lightning import loggers
#from pytorch_lightning.profilers import SimpleProfiler
#from pytorch_lightning.callbacks import EarlyStopping

import visde
import numpy as np
import torchsde
import torchdiffeq

torch.manual_seed(42)
torch.backends.cudnn.benchmark=True
print(f"cuDNN: {torch.backends.cudnn.is_available()}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data.pkl"

def get_dataloaders(n_win: int,
                    n_batch: int
) -> tuple[DataLoader, DataLoader]:
    with open(os.path.join(CURR_DIR, DATA_FILE), "rb") as f:
        data = pkl.load(f)

    train_data = visde.MultiEvenlySpacedTensors(data["train_mu"], data["train_t"], data["train_x"], data["train_f"], n_win)
    val_data = visde.MultiEvenlySpacedTensors(data["val_mu"], data["val_t"], data["val_x"], data["val_f"], n_win)

    train_sampler = visde.MultiTemporalSampler(train_data, n_batch, n_repeats=1)
    train_dataloader = DataLoader(
        train_data,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=train_sampler,
        pin_memory=True
    )
    val_sampler = visde.MultiTemporalSampler(val_data, n_batch, n_repeats=1)
    val_dataloader = DataLoader(
        val_data,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=val_sampler,
        pin_memory=True
    )

    return train_dataloader, val_dataloader

def dyn(t, x):
    visc = 0.005
    dim_x = int(np.sqrt(x.shape[-1]))
    x = x.reshape((dim_x, dim_x))

    h = 1.0/(dim_x - 1)
    dxdt = torch.zeros_like(x)

    dx11 = (x[2:dim_x, 1:dim_x-1] - x[0:dim_x-2, 1:dim_x-1])/(2*h)
    dx12 = (x[1:dim_x-1, 2:dim_x] - x[1:dim_x-1, 0:dim_x-2])/(2*h)
    dx21 = (x[2:dim_x, 1:dim_x-1] + x[0:dim_x-2, 1:dim_x-1] - 2*x[1:dim_x-1, 1:dim_x-1])/(h**2)
    dx22 = (x[1:dim_x-1, 2:dim_x] + x[1:dim_x-1, 0:dim_x-2] - 2*x[1:dim_x-1, 1:dim_x-1])/(h**2)
    dxdt[1:dim_x-1, 1:dim_x-1] = visc*(dx21 + dx22) - x[1:dim_x-1, 1:dim_x-1]*(dx11 + dx12)

    return dxdt.flatten()

class SDE(torch.nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, drift, dispersion):
        super().__init__()
        self.drift = drift
        self.dispersion = dispersion

    # Drift
    def f(self, t, z):
        t_ = t.reshape(1, 1).expand(z.shape[0], 1)
        return self.drift(torch.cat([z, t_], dim=-1))  # shape (n_batch, dim_z)

    # Diffusion
    def g(self, t, z):
        t_ = t.reshape(1, 1).expand(z.shape[0], 1)
        return self.dispersion(torch.cat([z, t_], dim=-1))  # shape (n_batch, dim_z, dim_z)

class niLES(pl.LightningModule):
    def __init__(self, dim_z_macro: int, lr: float, lr_sched_freq: int):
        super(niLES, self).__init__()
        self.coarse_grid_lin = int(round(np.sqrt(dim_z_macro)))
        self.lr = lr
        self.lr_sched_freq = lr_sched_freq

        self.latent_dim = 9
        self.n_zsamples = 128
        self.n_batch = 128
        self.enc_dim = self.coarse_grid_lin//8

        self.enc_mean = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(32*self.enc_dim*self.enc_dim, self.latent_dim)
        )

        self.enc_cov = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(32*self.enc_dim*self.enc_dim, self.latent_dim)
        )

        self.dec = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 32*self.enc_dim*self.enc_dim),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (32, self.enc_dim, self.enc_dim)),
            torch.nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3, padding=1, stride=2, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 3, padding=1, stride=2, output_padding=1)
        )

        self.drift = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim + 1, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, self.latent_dim)
        )

        self.dispersion = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim + 1, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, self.latent_dim**2),
            torch.nn.Unflatten(1, (self.latent_dim, self.latent_dim))
        )

        self.latent_sde = SDE(self.drift, self.dispersion)
    
    def encode(self, x):
        mean = self.enc_mean(x.unsqueeze(0)).squeeze(0)
        cov_vec = self.enc_cov(x.unsqueeze(0)).squeeze(0)
        cov = torch.diag_embed(cov_vec.exp())

        return mean, cov

    def training_step(self, batch, batch_idx):
        batch = [b.to(self.device) for b in batch]
        mu, t, x_win, x, f = batch

        t = t.flatten()
        n_tstep = t.shape[0]
        sigma = x.shape[-1] // self.coarse_grid_lin

        x_coarse = x[:, :, ::sigma, ::sigma]
        z0_mean, z0_cov = self.encode(x_coarse[0])
        z0 = torch.distributions.MultivariateNormal(z0_mean, z0_cov).sample((self.n_zsamples,))

        z_sde = torchsde.sdeint(self.latent_sde, z0, t, dt=1e-3, adaptive=False)
        x_int = torchdiffeq.odeint(dyn, x_coarse[0].flatten(), t, method="euler").unflatten(1, (1, self.coarse_grid_lin, self.coarse_grid_lin))
        x_clo = self.dec(z_sde.flatten(0, 1)).reshape(n_tstep, -1, 1, self.coarse_grid_lin, self.coarse_grid_lin).mean(1)
        x_rec = x_int + x_clo

        loss = torch.nn.functional.mse_loss(x_rec, x_coarse)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = [b.to(self.device) for b in batch]
        mu, t, x_win, x, f = batch

        t = t.flatten()
        n_tstep = t.shape[0]
        sigma = x.shape[-1] // self.coarse_grid_lin

        x_coarse = x[:, :, ::sigma, ::sigma]
        z0_mean, z0_cov = self.encode(x_coarse[0])
        z0 = torch.distributions.MultivariateNormal(z0_mean, z0_cov).sample((self.n_zsamples,))

        z_sde = torchsde.sdeint(self.latent_sde, z0, t, dt=1e-3, adaptive=False)
        x_int = torchdiffeq.odeint(dyn, x_coarse[0].flatten(), t, method="euler").unflatten(1, (1, self.coarse_grid_lin, self.coarse_grid_lin))
        x_clo = self.dec(z_sde.flatten(0, 1)).reshape(n_tstep, -1, 1, self.coarse_grid_lin, self.coarse_grid_lin).mean(1)
        x_rec = x_int + x_clo

        loss = torch.nn.functional.mse_loss(x_rec, x_coarse)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        gamma = np.exp(np.log(0.9) / self.lr_sched_freq)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "interval": "step",  # call scheduler after every train step
                "frequency": 1,
            },
        }

def main(overwrite: bool = False, dim_z_macro: int = 8**2, dim_z_micro: int = 0, max_epochs: int = 10, lr: float = 1e-3, lr_sched_freq: int = 100):
    n_win = 1
    print(f"CUDA: {torch.cuda.is_available()}")

    model = niLES(dim_z_macro, lr, lr_sched_freq)
    train_dataloader, val_dataloader = get_dataloaders(n_win, model.n_batch)

    version = "_".join([str(dim_z_macro), str(dim_z_micro), str(max_epochs), str(lr), str(lr_sched_freq)])
    if os.path.exists(os.path.join(CURR_DIR, "logs_visde", version)):
        if overwrite:
            print(f"Version {version} already exists. Overwriting...", flush=True)
            shutil.rmtree(os.path.join(CURR_DIR, "logs_visde", version))
        else:
            print(f"Version {version} already exists. Skipping...", flush=True)
            return
    
    tensorboard = loggers.TensorBoardLogger(CURR_DIR, name="logs_niles", version=version)
    #profiler = SimpleProfiler(dirpath=".", filename="perf_logs")

    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=max_epochs,
        logger=tensorboard,
        check_val_every_n_epoch=1,
        #profiler=profiler,
        #callbacks=[EarlyStopping(monitor="val/norm_rmse", mode="min")]
    )
    # ---------------------- training ---------------------- #
    trainer.fit(model, train_dataloader, val_dataloader)
    #print(profiler.summary())

if __name__ == "__main__":
    main()