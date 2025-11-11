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
from experiments_refac.shallow_water_2d.def_model import create_latent_sde
from experiments_refac.shallow_water_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.shallow_water_2d.load_data import load_data

plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

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

    dim_z = config.dim_z_macro + config.dim_z_micro
    shape_x = x.shape[2:]

    n_traj = mu.shape[0]
    n_win = 1
    n_batch = 64
    n_tsteps = t.shape[1]

    norm_rmse = np.zeros(n_traj)

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
        'method': 'srk',
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

    x_min = -1
    x_max = 1

    cmap = plt.get_cmap("ocean")

    for tsample in tsamples:
        z_i = zs[tsample].mean(0, keepdim=True)
        xr_i, _ = model.decoder(mu_i, z_i)
        x_smooth = model.decoder.macro_mean(mu_i, z_i[:, :config.dim_z_macro])
        x_resid = model.decoder.micro_mean(mu_i, z_i[:, config.dim_z_macro:])

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(x[i_traj, tsample, 0].cpu().detach().numpy(), cmap=cmap, vmin=x_min, vmax=x_max)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_true.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(xr_i[0, 0].cpu().detach().numpy(), cmap=cmap, vmin=x_min, vmax=x_max)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_recon.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(x_smooth[0, 0].cpu().detach().numpy(), cmap=cmap, vmin=x_min, vmax=x_max)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_smooth.pdf"))
        plt.close()

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(x_resid[0, 0].cpu().detach().numpy(), cmap=cmap, vmin=x_min, vmax=x_max)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{i_traj}_tsample_{tsample}_resid.pdf"))
        plt.close()


if __name__ == "__main__":
    main()