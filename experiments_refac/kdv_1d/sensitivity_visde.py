import torch

import os
import pathlib
from matplotlib import pyplot as plt
import numpy as np

import visde
from experiments_refac.kdv_1d.def_model import create_latent_sde
from experiments_refac.kdv_1d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Garamond",
    "font.size": 12
})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def finite_difference_dz(z):
    delta_x = 10/20

    # z has shape (n_batch, 5)
    dx = (z[:, 3] - z[:, 1])/(2*delta_x)
    #d2x = (z[:, 3] - 2*z[:, 2] + z[:, 1])/(delta_x**2)
    d3x = (z[:, 4] - 3*z[:, 3] + 3*z[:, 2] - z[:, 1])/(delta_x**3)
    #d3x = (z[:, 3] - 3*z[:, 2] + 3*z[:, 1] - z[:, 0])/(delta_x**3)
    #d3x = (z[:, 4] - 2*z[:, 3] + 2*z[:, 1] - z[:, 0])/(2*delta_x**3)
    #d4x = (z[:, 4] - 4*z[:, 3] + 6*z[:, 2] - 4*z[:, 1] + z[:, 0])/(delta_x**4)

    dxdt = -z[:, 2]*dx - 0.02*d3x

    return dxdt

def main(config: CaseConfig = DEFAULT_CONFIG) -> None:
    n_win = 1
    n_batch = 64
    #n_tsteps = t.shape[1]

    dummy_model = create_latent_sde(config, device)
    version = get_version_str(config)
    print(f"Version string: {version}")
    ckpt_dir = os.path.join(CURR_DIR, "logs_visde", version, "checkpoints")
    out_dir = os.path.join(CURR_DIR, "sensitivity_visde", version)

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

    #tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]
    
    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    i_traj = 0

    print(f"Trajectory {TRAIN_VAL_TEST} {i_traj}...", flush=True)

    #mu_i = mu[i_traj].unsqueeze(0)
    #mu_i_batch = mu_i.repeat((n_batch, 1))
    #t_i = t[i_traj]
    #x0_i = x[i_traj, :n_win, :].unsqueeze(0)
    #f_i = f[i_traj]

    fig, ax = plt.subplots(5, 1, figsize=(5, 6))

    z0 = torch.cat([torch.linspace(-1, 1, n_batch).unsqueeze(1), torch.zeros((n_batch, 4))], dim=1).to(device)
    with torch.no_grad():
        for i in range(5):
            dz0 = model.drift.macro_drift_net.fcnet_macro(torch.roll(z0, i, 1))
            fd_dz0 = finite_difference_dz(torch.roll(z0, i, 1))
            if i == 0:
                ax[i].plot(np.linspace(-1, 1, n_batch), fd_dz0.squeeze().cpu().detach().numpy(), label="From Finite Difference", color="black")
                ax2 = ax[i].twinx()
                ax2.plot(np.linspace(-1, 1, n_batch), dz0.squeeze().cpu().detach().numpy(), label="Learned", linestyle="--", color="red")
                handle1, label1 = ax[i].get_legend_handles_labels()
                handle2, label2 = ax2.get_legend_handles_labels()
                handles = handle1 + handle2
                labels = label1 + label2
            else:
                ax[i].plot(np.linspace(-1, 1, n_batch), fd_dz0.squeeze().cpu().detach().numpy(), color="black")
                ax2 = ax[i].twinx()
                ax2.plot(np.linspace(-1, 1, n_batch), dz0.squeeze().cpu().detach().numpy(), linestyle="--", color="red")

            ax2.tick_params(axis='y', labelcolor="red")
            low, high = ax2.get_ylim()
            bound = max(abs(low), abs(high))
            ax2.set_ylim(-bound, bound)
    
            if i < 4:
                ax[i].set_xticklabels([])
            else:
                ax[i].set_xlabel(r"Perturbation in $\zeta_{j-2}, \dots, \zeta_{j+2}$")

            ax[i].grid()
            if i == 0:
                ax[i].set_ylabel(r"$\widehat{f}_\theta(\zeta_{j-2}, ...)$")
            elif i == 1:
                ax[i].set_ylabel(r"$\widehat{f}_\theta(\zeta_{j-1}, ...)$")
            elif i == 2:
                ax[i].set_ylabel(r"$\widehat{f}_\theta(\zeta_j, ...)$")
            elif i == 3:
                ax[i].set_ylabel(r"$\widehat{f}_\theta(\zeta_{j+1}, ...)$")
            elif i == 4:
                ax[i].set_ylabel(r"$\widehat{f}_\theta(\zeta_{j+2}, ...)$")
   
    fig.legend(handles, labels, loc='upper center', ncol=2)
    fig.align_ylabels()
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(os.path.join(out_dir, "sensitivity.pdf"))
    fig.show()

if __name__ == "__main__":
    main()