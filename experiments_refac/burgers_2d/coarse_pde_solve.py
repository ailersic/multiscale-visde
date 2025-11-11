from jaxtyping import Float
import torch
from torch import Tensor
import pickle as pkl
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from experiments_refac.burgers_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.burgers_2d.load_data import load_data

# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
#plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
compplot_out_dir = os.path.join(CURR_DIR, "plot_dns")
pathlib.Path(compplot_out_dir).mkdir(parents=True, exist_ok=True)

TRAIN_VAL_TEST = "test"
nfe = 0

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(config: CaseConfig = DEFAULT_CONFIG) -> None:
    data = load_data(config.data_file)
    
    mu = data[f"{TRAIN_VAL_TEST}_mu"].cpu().detach().numpy()
    t = data[f"{TRAIN_VAL_TEST}_t"].cpu().detach().numpy()
    x = data[f"{TRAIN_VAL_TEST}_x"].cpu().detach().numpy()
    f = data[f"{TRAIN_VAL_TEST}_f"].cpu().detach().numpy()

    dim_x = np.prod(x.shape[2:])
    grid_x = x.shape[-1]
    sigma = int(np.sqrt(dim_x // config.dim_z_macro) + 0.5)
    grid_z = grid_x // sigma

    n_traj = mu.shape[0]
    n_tsteps = t.shape[1]

    z_pred = np.zeros((n_traj, n_tsteps, 1, grid_z, grid_z))
    for i in range(n_traj):
        z_pred[i, 0, :, :, :] = x[i, 0, :, ::sigma, ::sigma]

    global nfe
    for i in range(n_traj):
        sol = solve_ivp(dyn, [t[i, 0], t[i, -1]], z_pred[i, 0, :].flatten(), t_eval=t[i, :], method="LSODA")
        print(sol.message)
        z_pred[i] = np.transpose(sol.y).reshape(n_tsteps, 1, grid_z, grid_z)
        print(f"NFE cumulative: {nfe}")

    norm_err = np.zeros((n_traj, n_tsteps))

    #z_pred = z_pred.reshape(n_traj, n_tsteps, dim_z_macro)
    interp = RegularGridInterpolator(
        (np.linspace(0, 1, grid_z+1)[:-1], np.linspace(0, 1, grid_z+1)[:-1]),
        np.transpose(z_pred, (3, 4, 0, 1, 2)),
        method='cubic',
        bounds_error=False, fill_value=0
    )
    # meshgrid for interpolation
    x_grid = np.linspace(0, 1, grid_x+1)[:-1]
    y_grid = np.linspace(0, 1, grid_x+1)[:-1]
    X, Y = np.meshgrid(x_grid, y_grid)
    points = np.array([X.flatten(), Y.flatten()]).T
    x_pred = np.transpose(interp(points).reshape(grid_x, grid_x, n_traj, n_tsteps, 1), (2, 3, 4, 0, 1))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(x_pred[0, -1, 0], extent=(0, 1, 0, 1), origin='lower')
    axs[0].set_title('Predicted x at t=0')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[1].imshow(x[0, -1, 0], extent=(0, 1, 0, 1), origin='lower')
    axs[1].set_title('True x at t=0')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    plt.tight_layout()
    plt.savefig(os.path.join(CURR_DIR, 'predicted_x_t0.png'))
    plt.close()

    x_pred = x_pred.reshape(n_traj, n_tsteps, -1)
    x_true = x.reshape(n_traj, n_tsteps, dim_x)

    for i in range(n_traj):
        for j in range(n_tsteps):
            norm_err[i, j] = np.linalg.norm(x_pred[i, j] - x_true[i, j]) / np.linalg.norm(x_true[i, j])

        tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]
        cmap = plt.get_cmap("turbo")

        if i == 0:
            for j_plot, j_t in enumerate(tsamples):
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(x_pred[i, j_t].reshape(grid_x, grid_x), cmap=cmap, vmin=-0.22, vmax=1.06)
                plt.axis("off")
                fig.tight_layout()
                fig.savefig(os.path.join(compplot_out_dir, f"test_{i}_{j_plot}_pred.pdf"), format='pdf')
                fig.show()

        print(f"Trajectory {i}: mean error = {np.mean(norm_err[i])}, std error = {np.std(norm_err[i])}")
    print("Mean error across all trajectories:", np.mean(norm_err), "Std error:", np.std(norm_err))

def dyn(t: Float[Tensor, "n_traj"],
      x: Float[Tensor, "n_traj dim_x"]
) -> Float[Tensor, "n_traj dim_x"]:
    global nfe
    nfe += 1

    visc = 0.005
    dim_flat_x = x.shape[-1]
    dim_x = int(np.sqrt(dim_flat_x)+0.5)
    x = x.reshape(-1, dim_x, dim_x)
    h = 1.0/(dim_x - 1)
    dxdt = np.zeros_like(x)

    dx11 = (x[:, 2:dim_x, 1:dim_x-1] - x[:, 0:dim_x-2, 1:dim_x-1])/(2*h)
    dx12 = (x[:, 1:dim_x-1, 2:dim_x] - x[:, 1:dim_x-1, 0:dim_x-2])/(2*h)
    dx21 = (x[:, 2:dim_x, 1:dim_x-1] + x[:, 0:dim_x-2, 1:dim_x-1] - 2*x[:, 1:dim_x-1, 1:dim_x-1])/(h**2)
    dx22 = (x[:, 1:dim_x-1, 2:dim_x] + x[:, 1:dim_x-1, 0:dim_x-2] - 2*x[:, 1:dim_x-1, 1:dim_x-1])/(h**2)
    dxdt[:, 1:dim_x-1, 1:dim_x-1] = visc*(dx21 + dx22) - x[:, 1:dim_x-1, 1:dim_x-1]*(dx11 + dx12)

    return dxdt.reshape(-1, dim_flat_x)

if __name__ == "__main__":
    main()