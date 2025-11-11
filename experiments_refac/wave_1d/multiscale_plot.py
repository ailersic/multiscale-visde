import torch

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torchsde

import visde
from experiments_refac.wave_1d.def_model import create_latent_sde
from experiments_refac.wave_1d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.wave_1d.load_data import load_data

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "Garamond",
    "font.size": 16
})

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(config: CaseConfig = DEFAULT_CONFIG) -> None:
    data = load_data(config.data_file)
    
    mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    n_tsteps_trunc = 201

    t = t[:, :n_tsteps_trunc]
    x = x[:, :n_tsteps_trunc, :]
    f = f[:, :n_tsteps_trunc, :]

    dim_x = x.shape[-1]

    n_win = 1
    n_batch = 64
    n_tsteps = t.shape[1]

    dummy_model = create_latent_sde(config, device)
    version = get_version_str(config)
    print(f"Version string: {version}")
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

    print(f"Integrating SDE for trajectory {TRAIN_VAL_TEST} {i_traj}...", flush=True)

    sde_options = {
        'method': 'euler',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-3,
        'atol': 1e-5
    }

    mu_i = mu[i_traj].unsqueeze(0)
    t_i = t[i_traj]
    x0_i = x[i_traj, :n_win, :].unsqueeze(0)
    f_i = f[i_traj]

    z0_i = model.encoder.sample(n_batch, mu_i, x0_i)
    sde = visde.sde.SDE(model.drift, model.dispersion, mu_i, t_i, f_i)
    with torch.no_grad():
        zs = torchsde.sdeint(sde, z0_i, t_i, **sde_options)
    print("done", flush=True)

    for tsample in tsamples:
        z_i = zs[tsample].mean(0, keepdim=True)
        xr_i, _ = model.decoder(mu_i, z_i)
        x_smooth = model.decoder.macro_mean(mu_i, z_i[:, :config.dim_z_macro])
        x_resid = model.decoder.micro_mean(mu_i, z_i[:, config.dim_z_macro:])
        resid_mean = np.mean(x_resid[0, :].cpu().detach().numpy())
        x0_i = x[i_traj, tsample:(tsample+n_win), :].unsqueeze(0)

        x0_i, xr_i, x_smooth, x_resid = x0_i[:, :, 0], xr_i[:, 0], x_smooth[:, 0], x_resid[:, 0]

        fig, ax = plt.subplots(figsize=(7, 4))
        freq_domain = np.fft.rfftfreq(dim_x, d=1/dim_x)
        zfreq_domain = np.fft.rfftfreq(config.dim_z_macro, d=1/config.dim_z_macro)
        nyqz = np.max(zfreq_domain)
        #ax.title.set_text(f'Spectral decomposition at t={t_i[tsample]:.2f}')
        ax.add_line(plt.Line2D([nyqz, nyqz], [-1e6, 1e6], color='black', linewidth=2, linestyle='--'))
        ax.annotate('Macroscale\nNyquist Freq.', xy=(nyqz, 2e-3), ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.2'))
        ax.plot(freq_domain, torch.abs(torch.fft.rfft(x0_i[0, :])).cpu().detach().numpy(), linewidth=4, label=r"Observation", color='black')
        ax.plot(freq_domain, torch.abs(torch.fft.rfft(x_smooth[0, :])).cpu().detach().numpy(), linewidth=4, label=r"Prolonged Macro", color='blue', linestyle='-.')
        ax.plot(freq_domain, torch.abs(torch.fft.rfft(xr_i[0, :])).cpu().detach().numpy(), linewidth=4, label=r"Multiscale", color='red', linestyle='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([1e-3, 1000])
        ax.set_xlim([1, 100])
        ax.set_xlabel(r"Frequency")
        ax.set_ylabel(r"Magnitude")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_spectrum.pdf"), format='pdf')
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.linspace(0, 1, dim_x), x0_i[0, 0, :].cpu().detach().numpy(), linewidth=4, label="True", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.5, 1.2])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_true.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.linspace(0, 1, dim_x), xr_i[0, :].cpu().detach().numpy(), linewidth=4, label="Reconstruction", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.5, 1.2])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_recon.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.linspace(0, 1, dim_x), x_smooth[0, :].cpu().detach().numpy() + resid_mean, linewidth=4, label="Smooth", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.5, 1.2])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_smooth.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.linspace(0, 1, dim_x), x_resid[0, :].cpu().detach().numpy() - resid_mean, linewidth=4, label="Residual", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.5, 1.2])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_resid.pdf"))
        plt.close()


if __name__ == "__main__":
    main()