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
from experiments.burgers_2d.def_model import create_latent_sde

plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data_20_5_5.pkl"
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(dim_z_macro: int = 8*8,
         dim_z_micro: int = 5,
         max_epochs: int = 100,
         lr: float = 1e-4,
         lr_sched_freq: int = 2000,
         augment: bool = True,
) -> None:
    with open(os.path.join(CURR_DIR, DATA_FILE), "rb") as f:
        data = pkl.load(f)
    
    mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    dim_z = dim_z_macro + dim_z_micro
    shape_x = x.shape[2:]

    n_traj = mu.shape[0]
    n_win = 1
    n_batch = 64
    n_tsteps = t.shape[1]

    norm_rmse = np.zeros(n_traj)

    dummy_model = create_latent_sde(dim_z_macro, dim_z_micro, n_batch, n_win, lr, lr_sched_freq, DATA_FILE, device)
    if augment and dim_z_micro > 0:
        version = "_".join([str(dim_z_macro), str(dim_z_micro), str(max_epochs), str(lr), str(lr_sched_freq), "augment"])
    else:
        version = "_".join([str(dim_z_macro), str(dim_z_micro), str(max_epochs), str(lr), str(lr_sched_freq)])
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

    x_min = -0.22
    x_max = 1.06

    '''
    for j, j_t in enumerate(tsamples):
        mu_i = mu[i_traj].unsqueeze(0)
        t_i = t[i_traj]
        x0_i = x[i_traj, j_t:(j_t + n_win), :].unsqueeze(0)
        f_i = f[i_traj]

        z0_i, _ = model.encoder(mu_i, x0_i)
        xr_i, _ = model.decoder(mu_i, z0_i)
        x_smooth = model.decoder.decode_mean.decode_macro(z0_i[:, :dim_z_macro]).unflatten(1, shape_x)
        x_resid = model.decoder.decode_mean.decode_micro(z0_i[:, dim_z_macro:]).unflatten(1, shape_x)

        x_min = min(x_min, x[i_traj, j_t, 0].min(), x_smooth[0, 0].min(), x_resid[0, 0].min())
        x_max = max(x_max, x[i_traj, j_t, 0].max(), x_smooth[0, 0].max(), x_resid[0, 0].max())
    '''

    cmap = plt.get_cmap("turbo")

    for tsample in tsamples:
        mu_i = mu[i_traj].unsqueeze(0)
        t_i = t[i_traj]
        x0_i = x[i_traj, tsample:(tsample + n_win), :].unsqueeze(0)
        f_i = f[i_traj]

        z0_i, _ = model.encoder(mu_i, x0_i)
        xr_i, _ = model.decoder(mu_i, z0_i)
        x_smooth = model.decoder.decode_mean.decode_macro(z0_i[:, :dim_z_macro]).unflatten(1, shape_x)
        x_resid = model.decoder.decode_mean.decode_micro(z0_i[:, dim_z_macro:]).unflatten(1, shape_x)

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