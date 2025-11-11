import torch
from torch import Tensor
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
import numpy as np

import visde
from experiments_refac.wave_1d.def_model import create_latent_sde
from experiments_refac.wave_1d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.wave_1d.load_data import load_data

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Garamond",
    "font.size": 20
})
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

    n_tsteps_trunc = 201

    t = t[:, :n_tsteps_trunc]
    x = x[:, :n_tsteps_trunc, :]
    f = f[:, :n_tsteps_trunc, :]

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
        'method': 'euler',
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

    print("Decoding trajectory...", flush=True, end="")

    x_true = x[i_traj].cpu().detach().numpy()

    for j, j_t in enumerate(tsamples):
        xs = model.decoder.sample(n_batch_decoder, mu_i_batch, zs[j_t]).detach()
        x_mean = xs.mean(dim=0).cpu().detach().numpy()
        x_std = xs.std(dim=0).cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(5, 3))
        
        ax.plot(np.linspace(0, 1, dim_x), x_true[j_t, 0], color='black', linewidth=4)
        ax.plot(np.linspace(0, 1, dim_x), x_mean[0], color='red', linestyle='--', linewidth=4)
        ax.fill_between(np.linspace(0, 1, dim_x), x_mean[0] - x_std[0], x_mean[0] + x_std[0], alpha=0.3, color='red')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        ax.set_ylim(-0.5, 1.2)
        ax.set_xlim(0, 1)
        #if j == len(tsamples) - 1:
        #    ax.legend(["Observed", "Predicted"])
        #ax.set_title(f"{TRAIN_VAL_TEST} {i_traj} at t={t_i[j_t]:.2f}")
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")

        plt.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_{i_traj}_{j}_pred_vs_true.pdf"), format='pdf')
        fig.show()
    print("done", flush=True)

if __name__ == "__main__":
    main()