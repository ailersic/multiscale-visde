import torch
from torch import Tensor
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

import visde
from experiments.burgers_1d.def_model import create_latent_sde

plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data.pkl"
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(dim_z_macro: int = 20, dim_z_micro: int = 5, n_sigma: int = 3, max_epochs: int = 500, lr: float = 1e-3, lr_sched_freq: int = 1000):
    with open(os.path.join(CURR_DIR, DATA_FILE), "rb") as f:
        data = pkl.load(f)
    
    mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    dim_z = dim_z_macro + dim_z_micro
    dim_x = x.shape[-1]

    n_traj = mu.shape[0]
    n_win = 1
    n_batch = 64
    n_tsteps = t.shape[1]

    norm_rmse = np.zeros(n_traj)

    sde_options = {
        'method': 'srk',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-4,
        'atol': 1e-6
    }

    dummy_model = create_latent_sde(dim_z_macro, dim_z_micro, n_sigma, n_batch, n_win, lr, lr_sched_freq, DATA_FILE, device)
    version = "_".join([str(dim_z_macro), str(dim_z_micro), str(max_epochs), str(lr), str(lr_sched_freq), str(n_sigma)])
    ckpt_dir = os.path.join(CURR_DIR, "logs_visde", version, "checkpoints")
    out_dir = os.path.join(CURR_DIR, "msplot_visde", version)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(ckpt_dir):
        if file.endswith(".ckpt"):
            ckpt_file = file
    
    model = visde.LatentSDE.load_from_checkpoint(os.path.join(ckpt_dir, ckpt_file),
                                                 config=dummy_model.config,
                                                 encoder=dummy_model.encoder,
                                                 decoder=dummy_model.decoder,
                                                 drift=dummy_model.drift,
                                                 dispersion=dummy_model.dispersion,
                                                 loglikelihood=dummy_model.loglikelihood,
                                                 latentvar=dummy_model.latentvar).to(device)
    model.eval()
    model.encoder.resample_params()
    model.decoder.resample_params()
    model.drift.resample_params()
    model.dispersion.resample_params()

    tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]
    i_traj = 0

    print(f"Not integrating SDE for trajectory {TRAIN_VAL_TEST} {i_traj}...", flush=True)

    for tsample in tsamples:
        mu_i = mu[i_traj].unsqueeze(0)
        t_i = t[i_traj]
        x0_i = x[i_traj, tsample:(tsample + n_win), :].unsqueeze(0)
        f_i = f[i_traj]

        z0_i, _ = model.encoder(mu_i, x0_i)
        xr_i, _ = model.decoder(mu_i, z0_i)
        x_smooth = model.decoder.decode_mean.decode_macro(z0_i[:, :dim_z_macro])
        x_resid = model.decoder.decode_mean.decode_micro(z0_i[:, dim_z_macro:])

        fig, ax = plt.subplots(figsize=(7, 4))
        freq_domain = np.fft.rfftfreq(dim_x, d=1/dim_x)
        zfreq_domain = np.fft.rfftfreq(dim_z_macro, d=1/dim_z_macro)
        nyqz = np.max(zfreq_domain)
        ax.title.set_text(f"Spectral decomposition at t={t_i[tsample]:.2f}")
        ax.add_line(plt.Line2D([nyqz, nyqz], [-1e6, 1e6], color='black', linewidth=2, linestyle='--'))
        ax.annotate('Macroscale\nNyquist Freq.', xy=(nyqz, 2e-4), ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.2'))
        ax.plot(freq_domain, torch.abs(torch.fft.rfft(x0_i[0, 0, :])).cpu().detach().numpy(), linewidth=4, label="True", color='black')
        ax.plot(freq_domain, torch.abs(torch.fft.rfft(x_smooth[0, :])).cpu().detach().numpy(), linewidth=4, label="Macro", color='blue')
        ax.plot(freq_domain, torch.abs(torch.fft.rfft(x_smooth[0, :] + x_resid[0, :])).cpu().detach().numpy(), linewidth=4, label="Multi", color='red')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([1e-4, 1e2])
        ax.set_xlim([1, 100])
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_spectrum.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.linspace(0, 1, dim_x), x0_i[0, 0, :].cpu().detach().numpy(), linewidth=4, label="True", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.2, 0.5])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_true.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.linspace(0, 1, dim_x), xr_i[0, :].cpu().detach().numpy(), linewidth=4, label="Reconstruction", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.2, 0.5])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_recon.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.linspace(0, 1, dim_x), x_smooth[0, :].cpu().detach().numpy(), linewidth=4, label="Smooth", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.2, 0.5])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_smooth.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.linspace(0, 1, dim_x), x_resid[0, :].cpu().detach().numpy(), linewidth=4, label="Residual", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.2, 0.5])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_resid.pdf"))
        plt.close()


if __name__ == "__main__":
    main()