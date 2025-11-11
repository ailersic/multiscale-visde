import torch
from torch import Tensor
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from matplotlib import animation

import visde
from experiments_refac.shallow_water_2d.def_model import create_latent_sde
from experiments_refac.shallow_water_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.shallow_water_2d.load_data import load_data

plt.rcParams.update({'font.size': 20})
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

    i_traj = 0
    n_traj = mu.shape[0]
    n_win = 1
    n_batch_encoder = 16
    n_batch_decoder = 16
    n_tsteps = t.shape[1] # change value manually if you want a shorter animation

    t = t[:, :n_tsteps]
    x = x[:, :n_tsteps, :, :, :]
    f = f[:, :n_tsteps, :]

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
    out_dir = os.path.join(CURR_DIR, "postproc_visde", version)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Trajectory {i_traj+1}/{n_traj} from {TRAIN_VAL_TEST} set", flush=True)

    traj_dir = os.path.join(out_dir, f"{TRAIN_VAL_TEST}_traj_{i_traj}")
    pathlib.Path(traj_dir).mkdir(parents=True, exist_ok=True)

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

    fig, axgrid = plt.subplots(figsize=(12, 4), nrows=1, ncols=8, width_ratios=[1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2], layout="constrained", squeeze=False)
    
    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    print(f"Integrating SDE for trajectory {TRAIN_VAL_TEST} {i_traj}...", flush=True)

    mu_i = mu[i_traj].unsqueeze(0)
    mu_i_batch = mu_i.repeat((n_batch_encoder, 1))
    t_i = t[i_traj]
    x0_i = x[i_traj, :n_win, :].unsqueeze(0)
    f_i = f[i_traj]

    z0_i = model.encoder.sample(n_batch_encoder, mu_i, x0_i)
    sde = visde.sde.SDE(model.drift, model.dispersion, mu_i, t_i, f_i)
    with torch.no_grad():
        zs = torchsde.sdeint(sde, z0_i, t_i, **sde_options)
    print("done", flush=True)

    assert isinstance(zs, Tensor), "zs is expected to be a single tensor"

    x_true = x[i_traj].cpu().detach().numpy()

    cmap = 'ocean'

    x_min = -1#np.min(x_true[:, 0])
    x_max = 1#np.max(x_true[:, 0])
    im1 = axgrid[0, 0].imshow(x_true[0, 0], cmap=cmap, vmin=x_min, vmax=x_max)
    fig.colorbar(im1, ax=axgrid[0, 7], aspect=10, fraction=0.4, ticks=[-1, 0, 1])

    def update(j):
        if j % 10 == 0:
            print(f"Frame {j}/{n_tsteps}", flush=True)
        # j is time index

        for axcol in axgrid:
            for ax in axcol:
                ax.clear()

        xs = model.decoder.sample(n_batch_decoder, mu_i_batch, zs[j]).detach()
        x_mean = xs.mean(dim=0).cpu().detach().numpy()
        #x_std = xs.std(dim=0).cpu().detach().numpy()

        x_macro = model.decoder.macro_mean(mu_i_batch, zs[j, :, :config.dim_z_macro]).mean(dim=0).cpu().detach().numpy()

        axgrid[0, 0].imshow(x_true[j, 0], cmap=cmap, vmin=x_min, vmax=x_max)
        axgrid[0, 2].imshow(x_mean[0], cmap=cmap, vmin=x_min, vmax=x_max)
        axgrid[0, 4].imshow(x_macro[0], cmap=cmap, vmin=x_min, vmax=x_max)

        if config.dim_z_micro > 0:
            x_micro = model.decoder.micro_mean(mu_i_batch, zs[j, :, config.dim_z_macro:]).mean(dim=0).cpu().detach().numpy()
            axgrid[0, 6].imshow(x_micro[0], cmap=cmap, vmin=x_min, vmax=x_max)
        else:
            axgrid[0, 6].annotate("N/A", xy=(0.5, 0.5), fontsize=20, ha="center", va="center")
            axgrid[0, 6].set_xlim(0, 1)
            axgrid[0, 6].set_ylim(0, 1)

        axgrid[0, 1].text(-0.4, 0, r"$\approx$", fontsize=36, ha="center", va="center")
        axgrid[0, 1].set_xlim(-1, 1)
        axgrid[0, 1].set_ylim(-1, 1)

        axgrid[0, 3].text(-0.4, 0, r"$=$", fontsize=36, ha="center", va="center")
        axgrid[0, 3].set_xlim(-1, 1)
        axgrid[0, 3].set_ylim(-1, 1)

        axgrid[0, 5].text(-0.1, 0, r"$+$", fontsize=36, ha="center", va="center")
        axgrid[0, 5].set_xlim(-1, 1)
        axgrid[0, 5].set_ylim(-1, 1)

        for i in [0, 2, 4, 6]:
            axgrid[0, i].set_xticks([])
            axgrid[0, i].set_yticks([])
            axgrid[0, i].set_aspect("equal")

        for i in [1, 3, 5, 7]:
            axgrid[0, i].axis("off")

        #axgrid[0].set_ylabel(f"t={t[i_traj, j]:.2f}")
        axgrid[0, 0].set_title("Ground Truth")
        axgrid[0, 0].set_xlabel("Resolved\nat hi-res")
        axgrid[0, 2].set_title("Prediction")
        axgrid[0, 2].set_xlabel("Rendered\nat hi-res")
        axgrid[0, 4].set_title("Macroscale")
        axgrid[0, 4].set_xlabel("Captures features\non lo-res grid")
        axgrid[0, 6].set_title("Microscale")
        axgrid[0, 6].set_xlabel("Captures sub-\ngrid-scale features")
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_tsteps, interval=30)
    ani.save(filename=os.path.join(traj_dir, "anim.gif"), writer="pillow")
    fig.show()

if __name__ == "__main__":
    main()