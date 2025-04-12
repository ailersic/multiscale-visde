import torch
from torch import Tensor
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
from jaxtyping import Float
from scipy.integrate import solve_ivp

import visde
from experiments.kdv_1d.def_model import create_latent_sde

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Garamond",
    "font.size": 20
})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data.pkl"
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def dyn(#mu: Float[Tensor, "n_traj dim_mu"],
        t: Float[Tensor, ""],
        x: Float[Tensor, "dim_x"],
        #f: Float[Tensor, "n_traj dim_f"]
) -> Float[Tensor, "dim_x"]:
    d = 0.02#mu[:, 0:1]

    dim_x = x.shape[-1]
    delta_x = 10/dim_x # not dim_x - 1 because boundary conditions are periodic

    ip2 = (np.arange(dim_x) + 2) % dim_x
    ip1 = (np.arange(dim_x) + 1) % dim_x
    im1 = (np.arange(dim_x) - 1) % dim_x
    im2 = (np.arange(dim_x) - 2) % dim_x

    xp2 = x[ip2]
    xp1 = x[ip1]
    xm1 = x[im1]
    xm2 = x[im2]
    dx = (xp1 - xm1)/(2*delta_x)
    #d3x = (xp2 - 2*xp1 + 2*xm1 - xp2)/(2*delta_x**3)
    d3x = (xp2 - 3*xp1 + 3*x - xm1)/(delta_x**3)

    dxdt = -x*dx - d*d3x

    return dxdt

def main(dim_z_macro: int = 25, dim_z_micro: int = 0, n_sigma = 3, max_epochs: int = 500, lr: float = 1e-3, lr_sched_freq: int = 1000):
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
    n_batch = 128
    n_batch_decoder = 32
    n_tsteps = t.shape[1]

    norm_rmse = np.zeros(n_traj)

    sde_options = {
        'method': 'euler',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-3,
        'atol': 1e-5
    }

    dummy_model = create_latent_sde(dim_z_macro, dim_z_micro, n_sigma, n_batch, n_win, lr, lr_sched_freq, DATA_FILE, device)
    version = "_".join([str(dim_z_macro), str(dim_z_micro), str(max_epochs), str(lr), str(lr_sched_freq), str(n_sigma)])
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

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]
    
    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    for i_traj in range(n_traj):
        print(f"Integrating SDE for trajectory {TRAIN_VAL_TEST} {i_traj}...", flush=True)

        mu_i = mu[i_traj].unsqueeze(0)
        mu_i_batch = mu_i.repeat((n_batch, 1))
        t_i = t[i_traj]
        x0_i_np = x[i_traj, 0, :].cpu().detach().numpy()
        f_i = f[i_traj]
        sigma = dim_x // dim_z_macro

        ###
        sol = solve_ivp(dyn, [t_i[0], t_i[-1]], x0_i_np[::sigma], t_eval=t_i.cpu().detach().numpy(), method="LSODA")
        print(sol.message)
        xfd = torch.tensor(sol.y).T
        print(xfd.shape)
        ###

        x0_i = x[i_traj, :n_win, :].unsqueeze(0)
        
        z0_i = model.encoder.sample(n_batch, mu_i, x0_i)
        sde = visde.sde.SDE(model.drift, model.dispersion, mu_i, t_i, f_i)
        with torch.no_grad():
            zs = torchsde.sdeint(sde, z0_i, t_i, **sde_options)
        print("done", flush=True)

        x_true = x[i_traj].cpu().detach().numpy()

        for j, j_t in enumerate(tsamples):
            xs = model.decoder.sample(n_batch_decoder, mu_i_batch, zs[j_t]).detach()
            x_mean = xs.mean(dim=0).cpu().detach().numpy()
            x_std = xs.std(dim=0).cpu().detach().numpy()

            fig, ax = plt.subplots(figsize=(5, 4))
            
            ax.plot(np.linspace(0, 1, dim_x+1)[:dim_x], x_true[j_t], color='black', linewidth=4)
            ax.plot(np.linspace(0, 1, dim_x+1)[:dim_x], x_mean, color='red', linestyle='--', linewidth=4)
            ax.plot(np.linspace(0, 1, dim_z_macro+1)[:dim_z_macro], xfd[j_t], color='blue', linestyle=':', linewidth=4)
            ax.fill_between(np.linspace(0, 1, dim_x+1)[:dim_x], x_mean - x_std, x_mean + x_std, alpha=0.3, color='red')
            ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            ax.set_ylim(-2, 2)
            ax.set_xlim(0, 1)
            if j == len(tsamples) - 1:
                ax.legend(["Observed", "Closure", "DNS"])
            #ax.set_title(f"{TRAIN_VAL_TEST} {i_traj} at t={t_i[j_t]:.2f}")
            #ax.set_xlabel("x")
            #ax.set_ylabel("y")

            plt.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_{i_traj}_{j}_pred_fd_true.pdf"), format='pdf')
            fig.show()

if __name__ == "__main__":
    main()