from jaxtyping import Float
import torch
from torch import Tensor
import pickle as pkl
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from experiments_refac.kdv_1d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.kdv_1d.load_data import load_data

# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
#plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

TRAIN_VAL_TEST = "test"
nfe = 0

compplot_out_dir = os.path.join(CURR_DIR, "plot_dns")
pathlib.Path(compplot_out_dir).mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(config: CaseConfig = DEFAULT_CONFIG) -> None:
    data = load_data(config.data_file)
    
    mu = data[f"{TRAIN_VAL_TEST}_mu"].cpu().detach().numpy()
    t = data[f"{TRAIN_VAL_TEST}_t"].cpu().detach().numpy()
    x = data[f"{TRAIN_VAL_TEST}_x"].squeeze(2).cpu().detach().numpy()
    f = data[f"{TRAIN_VAL_TEST}_f"].cpu().detach().numpy()

    dim_x = x.shape[-1]
    sigma = dim_x // config.dim_z_macro

    n_traj = mu.shape[0]
    n_tsteps = t.shape[1]

    z_pred = np.zeros((n_traj, n_tsteps, config.dim_z_macro))
    for i in range(n_traj):
        z_pred[i, 0, :] = x[i, 0, ::sigma]

    global nfe
    for i in range(n_traj):
        sol = solve_ivp(dyn, [t[i, 0], t[i, -1]], z_pred[i, 0, :], t_eval=t[i, :], method="LSODA")
        print(sol.message)
        z_pred[i] = np.transpose(sol.y)
        print(f"NFE cumulative: {nfe}")

    norm_err = np.zeros((n_traj, n_tsteps))

    x_pred = np.zeros((n_traj, n_tsteps, dim_x))
    # cubic interpolation from z_pred to x_pred
    for i in range(n_traj):
        cs = CubicSpline(np.linspace(0, 1, config.dim_z_macro+1), np.concatenate([z_pred[i, :, :], z_pred[i, :, 0:1]], axis=1), axis=1)
        x_pred[i] = cs(np.linspace(0, 1, dim_x+1)[:-1])

        tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]

        if i == 0:
            for j_plot, j_t in enumerate(tsamples):
                fig, ax = plt.subplots(figsize=(5, 3))

                ax.plot(np.linspace(0, 1, x[i, j_t].shape[0]), x[i, j_t], color='black', linewidth=4)
                ax.plot(np.linspace(0, 1, x_pred[i, j_t].shape[0]), x_pred[i, j_t], color='red', linestyle='--', linewidth=4)
                ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
                ax.set_ylim(-2, 2)
                ax.set_xlim(0, 1)

                plt.axis("off")
                fig.tight_layout()
                fig.savefig(os.path.join(compplot_out_dir, f"test_{i}_{j_plot}_pred_vs_true.pdf"), format='pdf')
                fig.show()

    plt.plot(x_pred[0, 0, :], label='Predicted x at t=0')
    plt.plot(x[0, 0, :], label='True x at t=0')
    plt.legend()
    plt.title('Comparison of predicted and true x at t=0')
    plt.xlabel('Spatial coordinate')
    plt.ylabel('Value')
    plt.savefig(os.path.join(CURR_DIR, 'predicted_x_t0.png'))
    plt.close()

    plt.plot(x_pred[0, n_tsteps//2, :], label='Predicted x at t=0.5')
    plt.plot(x[0, n_tsteps//2, :], label='True x at t=0.5')
    plt.legend()
    plt.title('Comparison of predicted and true x at t=0.5')
    plt.xlabel('Spatial coordinate')
    plt.ylabel('Value')
    plt.savefig(os.path.join(CURR_DIR, 'predicted_x_t05.png'))
    plt.close()

    x_true = x

    for i in range(n_traj):
        for j in range(n_tsteps):
            norm_err[i, j] = np.linalg.norm(x_pred[i, j, :] - x_true[i, j, :]) / np.linalg.norm(x_true[i, j, :])

        print(f"Trajectory {i}: mean error = {np.mean(norm_err[i])}, std error = {np.std(norm_err[i])}")

    print("Mean error across all trajectories:", np.mean(norm_err), "Std error:", np.std(norm_err))
    print(f"Avg NFE per traj: {nfe/n_traj}")

def dyn(t: Float[Tensor, ""],
        x: Float[Tensor, "dim_x"],
) -> Float[Tensor, "dim_x"]:
    global nfe
    nfe += 1

    d = 0.02

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
    d3x = (xp2 - 3*xp1 + 3*x - xm1)/(delta_x**3)

    dxdt = -x*dx - d*d3x

    return dxdt

if __name__ == "__main__":
    main()