import torch
from torch import Tensor
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from matplotlib import animation

import visde
from experiments_refac.cylinder_2d.def_model import create_latent_sde
from experiments_refac.cylinder_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.cylinder_2d.load_data import load_data

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

    fig, axgrid = plt.subplots(figsize=(24, 4), nrows=2, ncols=8, width_ratios=[1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2], layout="constrained", squeeze=False)
    
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

    cmap = 'turbo'

    x_min = -1#np.min(x_true[:, 0])
    x_max = 1.5#np.max(x_true[:, 0])
    im1 = axgrid[0, 0].imshow(x_true[0, 0], cmap=cmap, vmin=x_min, vmax=x_max)
    fig.colorbar(im1, ax=axgrid[0, 7], aspect=10, fraction=0.15, ticks=[-1, 0, 1])

    y_min = -1#np.min(x_true[:, 0])
    y_max = 1.5#np.max(x_true[:, 0])
    im1b = axgrid[1, 0].imshow(x_true[0, 1], cmap=cmap, vmin=y_min, vmax=y_max)
    fig.colorbar(im1b, ax=axgrid[1, 7], aspect=10, fraction=0.15, ticks=[-1, 0, 1])

    def update(j):
        if j % 10 == 0:
            print(f"Frame {j}/{n_tsteps}", flush=True)
        # j is time index

        for axcol in axgrid:
            for ax in axcol:
                ax.clear()

        xs = model.decoder.sample(n_batch_decoder, mu_i_batch, zs[j]).detach()
        x_mean = xs.mean(dim=0).cpu().detach().numpy()
        x_std = xs.std(dim=0).cpu().detach().numpy()

        x_macro = model.decoder.macro_mean(mu_i_batch, zs[j, :, :config.dim_z_macro]).mean(dim=0).cpu().detach().numpy()

        im1 = axgrid[0, 0].imshow(x_true[j, 0], cmap=cmap, vmin=x_min, vmax=x_max)
        im2 = axgrid[0, 2].imshow(x_mean[0], cmap=cmap, vmin=x_min, vmax=x_max)
        im3 = axgrid[0, 4].imshow(x_macro[0], cmap=cmap, vmin=x_min, vmax=x_max)

        im1b = axgrid[1, 0].imshow(x_true[j, 1], cmap=cmap, vmin=y_min, vmax=y_max)
        axgrid[1, 2].imshow(x_mean[1], cmap=cmap, vmin=y_min, vmax=y_max)
        axgrid[1, 4].imshow(x_macro[1], cmap=cmap, vmin=y_min, vmax=y_max)

        if config.dim_z_micro > 0:
            x_micro = model.decoder.micro_mean(mu_i_batch, zs[j, :, config.dim_z_macro:]).mean(dim=0).cpu().detach().numpy()
            im4 = axgrid[0, 6].imshow(x_micro[0], cmap=cmap, vmin=x_min, vmax=x_max)
            axgrid[1, 6].imshow(x_micro[1], cmap=cmap, vmin=y_min, vmax=y_max)
        else:
            im4 = axgrid[0, 6].annotate("N/A", xy=(2.0, 0.5), fontsize=20, ha="center", va="center")
            axgrid[1, 6].annotate("N/A", xy=(2.0, 0.5), fontsize=20, ha="center", va="center")

            axgrid[0, 6].set_xlim(0, 4)
            axgrid[0, 6].set_ylim(0, 1)

            axgrid[1, 6].set_xlim(0, 4)
            axgrid[1, 6].set_ylim(0, 1)

        for i in [0, 1]:
            axgrid[i, 1].text(-0.2, 0, r"$\approx$", fontsize=48, ha="center", va="center")
            axgrid[i, 1].set_xlim(-1, 1)
            axgrid[i, 1].set_ylim(-1, 1)

            axgrid[i, 3].text(-0.2, 0, r"$=$", fontsize=48, ha="center", va="center")
            axgrid[i, 3].set_xlim(-1, 1)
            axgrid[i, 3].set_ylim(-1, 1)

            axgrid[i, 5].text(-0.1, 0, r"$+$", fontsize=48, ha="center", va="center")
            axgrid[i, 5].set_xlim(-1, 1)
            axgrid[i, 5].set_ylim(-1, 1)

        for i in [0, 2, 4, 6]:
            axgrid[0, i].set_xticks([])
            axgrid[0, i].set_yticks([])
            axgrid[0, i].set_aspect("equal")
            axgrid[1, i].set_xticks([])
            axgrid[1, i].set_yticks([])
            axgrid[1, i].set_aspect("equal")

            #axgrid[0, i].add_patch(Ellipse(xy=(40.0, 40.0), width=10, height=10, angle=0, edgecolor='black', facecolor='white', lw=1))
            #axgrid[1, i].add_patch(Ellipse(xy=(40.0, 40.0), width=10, height=10, angle=0, edgecolor='black', facecolor='white', lw=1))

        for i in [1, 3, 5, 7]:
            axgrid[0, i].axis("off")
            axgrid[1, i].axis("off")

        #axgrid[0].set_ylabel(f"t={t[i_traj, j]:.2f}")
        axgrid[0, 0].set_ylabel("x Velocity")
        axgrid[1, 0].set_ylabel("y Velocity")
        axgrid[0, 0].set_title("True Solution")
        axgrid[0, 2].set_title("Prediction Mean")
        axgrid[0, 4].set_title("Macroscale")
        axgrid[0, 6].set_title("Microscale")
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_tsteps, interval=30)
    ani.save(filename=os.path.join(traj_dir, "anim.gif"), writer="pillow")
    fig.show()

if __name__ == "__main__":
    main()