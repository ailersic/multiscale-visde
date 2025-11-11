import torch

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
import numpy as np

import visde
from experiments_refac.kdv_1d.def_model import create_latent_sde
from experiments_refac.kdv_1d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.kdv_1d.load_data import load_data

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

    dim_x = x.shape[-1]

    n_win = 1
    n_batch = 64
    n_tsteps = t.shape[1]

    dummy_model = create_latent_sde(config, device)
    version = get_version_str(config)
    print(f"Version string: {version}")
    ckpt_dir = os.path.join(CURR_DIR, "logs_visde", version, "checkpoints")
    out_dir = os.path.join(CURR_DIR, "macroplot_visde", version)

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

    tsamples = [n_tsteps-1]
    i_traj = 0

    print(f"Not integrating SDE for trajectory {TRAIN_VAL_TEST} {i_traj}...", flush=True)

    for tsample in tsamples:
        mu_i = mu[i_traj].unsqueeze(0)
        #t_i = t[i_traj]
        x0_i = x[i_traj, tsample:(tsample + n_win), :].unsqueeze(0)
        #f_i = f[i_traj]

        x_smooth = model.encoder.smooth_net(mu_i, x0_i)
        z0_i, _ = model.encoder(mu_i, x0_i)
        #xr_i, _ = model.decoder(mu_i, z0_i)
        x_deconv = model.decoder.macro_mean(mu_i, z0_i[:, :config.dim_z_macro])
        #x_resid = model.decoder.decode_mean.decode_micro(z0_i[:, dim_z_macro:])

        scale = (torch.norm(x0_i.flatten())/torch.norm(x_smooth.flatten())).item()

        x_mesh = np.linspace(0, 1, dim_x+1)[:-1]
        z_mesh = np.linspace(0, 1, config.dim_z_macro+1)[:-1]

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x_mesh, x0_i[0, 0, 0, :].cpu().detach().numpy(), linewidth=4, label="True", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-2.0, 2.0])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_true.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x_mesh, x_smooth[0, 0, 0, :].cpu().detach().numpy(), linewidth=4, label="Smooth", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-2.0/scale, 2.0/scale])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_smooth.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(z_mesh, z0_i[0, :config.dim_z_macro].cpu().detach().numpy(), linewidth=4, label="Macro", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-2.0/scale, 2.0/scale])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_macro.pdf"))

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x_mesh, x_deconv[0, 0, :].cpu().detach().numpy(), linewidth=4, label="Deconv", color='black')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        ax.set_xlim([0, 1])
        ax.set_ylim([-2.0, 2.0])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_deconv.pdf"))
        plt.close()


if __name__ == "__main__":
    main()