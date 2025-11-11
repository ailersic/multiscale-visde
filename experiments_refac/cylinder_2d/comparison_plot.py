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
from experiments_refac.cylinder_2d.def_model import create_latent_sde
from experiments_refac.cylinder_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.cylinder_2d.load_data import load_data

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data.pkl"
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
    dim_x = x.shape[-1]

    n_traj = mu.shape[0]
    n_win = 1
    n_batch = 64
    n_batch_decoder = 128
    n_tsteps = t.shape[1]

    i_traj = 0

    norm_rmse = np.zeros(n_traj)

    sde_options = {
        'method': 'srk',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-3,
        'atol': 1e-5
    }

    dummy_model = create_latent_sde(config, device)
    version = get_version_str(config)
    print(f"Version string: {version}")
    ckpt_dir = os.path.join(CURR_DIR, "logs_visde", version, "checkpoints")
    out_dir = os.path.join(CURR_DIR, "plot_visde", version)

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
    
    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)

    print(f"Integrating SDE for trajectory {TRAIN_VAL_TEST} {i_traj}...", flush=True)

    mu_i = mu[i_traj].unsqueeze(0)
    mu_i_batch = mu_i.repeat((n_batch, 1))
    t_i = t[i_traj]
    x0_i = x[i_traj, :n_win, :].unsqueeze(0)
    f_i = f[i_traj]

    z0_i = model.encoder.sample(n_batch, mu_i, x0_i)
    sde = visde.sde.SDE(model.drift, model.dispersion, mu_i, t_i, f_i)
    with torch.no_grad():
        zs = torchsde.sdeint(sde, z0_i, t_i, **sde_options)
    print("done", flush=True)

    assert isinstance(zs, Tensor), "zs is expected to be a single tensor"

    sqerr = np.zeros(n_tsteps)
    norm_sqerr = np.zeros(n_tsteps)
    aenc_sqerr = np.zeros(n_tsteps)
    aenc_norm_sqerr = np.zeros(n_tsteps)

    z_i = torch.zeros(n_tsteps, 1, dim_z).to(device)

    print("Decoding trajectory...", flush=True, end="")
    for j_t in range(n_tsteps):
        if j_t % 100 == 0:
            print(f"{j_t}...", flush=True, end="")

        xs = model.decoder.sample(n_batch_decoder, mu_i_batch, zs[j_t]).detach()
        x_mean = xs.mean(dim=0)
        x_err = x_mean - x[i_traj, j_t]

        sqerr[j_t] = x_err.pow(2).sum().item()
        norm_sqerr[j_t] = sqerr[j_t] / x[i_traj, j_t].pow(2).sum().item()

        z_i[j_t], _ = model.encoder(mu_i, x[i_traj, j_t:(j_t+n_win)].unsqueeze(0))
        x_rec_ij, _ = model.decoder(mu_i, z_i[j_t])
        aenc_err = x_rec_ij - x[i_traj, j_t]

        aenc_sqerr[j_t] = aenc_err.pow(2).sum().item()
        aenc_norm_sqerr[j_t] = aenc_sqerr[j_t] / x[i_traj, j_t].pow(2).sum().item()
    print("done", flush=True)

    norm_rmse[i_traj] = np.sqrt(np.mean(norm_sqerr))
    
    print(f"Mean Normalized RMSE: {norm_rmse[i_traj]}", flush=True)

    x_true = x[i_traj].cpu().detach().numpy()

    cmap = plt.get_cmap("turbo")

    for j, j_t in enumerate(tsamples):
        xs = model.decoder.sample(n_batch_decoder, mu_i_batch, zs[j_t]).detach()
        x_mean = xs.mean(dim=0).cpu().detach().numpy()
        x_std = xs.std(dim=0).cpu().detach().numpy()

        print(np.max(np.abs(x_true[j_t, 0] - x_mean[0])), np.max(x_std[0]), flush=True)
        for chan in range(2):
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.imshow(x_true[j_t, chan], cmap=cmap, vmin=-1.0, vmax=1.5)
            plt.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_{i_traj}_{j}_true_{chan}.pdf"), format='pdf')
            fig.show()

            fig, ax = plt.subplots(figsize=(5, 2))
            ax.imshow(x_mean[chan], cmap=cmap, vmin=-1.0, vmax=1.5)
            plt.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_{i_traj}_{j}_pred_{chan}.pdf"), format='pdf')
            fig.show()

            fig, ax = plt.subplots(figsize=(5, 2))
            ax.imshow(np.abs(x_mean[chan] - x_true[j_t, chan]), cmap="afmhot", vmin=0.0, vmax=0.14)
            plt.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_{i_traj}_{j}_err_{chan}.pdf"), format='pdf')
            fig.show()

            fig, ax = plt.subplots(figsize=(5, 2))
            ax.imshow(x_std[chan], cmap="afmhot", vmin=0.0, vmax=0.14)
            plt.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_{i_traj}_{j}_std_{chan}.pdf"), format='pdf')
            fig.show()

    print(f"Normalized RMSE Mean: {np.mean(norm_rmse)}, Std Dev: {np.std(norm_rmse)}", flush=True)

if __name__ == "__main__":
    main()