import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import pysindy as ps
import pickle as pkl
from scipy.integrate import solve_ivp

from experiments_refac.shallow_water_2d.sindy_2step.def_model import create_sindy_model
from experiments_refac.shallow_water_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.shallow_water_2d.load_data import load_data

#plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
TRAIN_VAL_TEST = "test"

compplot_out_dir = os.path.join(CURR_DIR, "..", "plot_sindy")
pathlib.Path(compplot_out_dir).mkdir(parents=True, exist_ok=True)

def main(dim_z: int = 69,
        threshold: float = 1e-2,
        degree: int = 1,
        config: CaseConfig = DEFAULT_CONFIG,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(config.data_file)

    mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    n_tsteps_trunc = -1

    t = t[:, :n_tsteps_trunc]
    x = x[:, :n_tsteps_trunc, :]
    f = f[:, :n_tsteps_trunc, :]

    shape_x = x.shape[2:]
    dim_x = np.prod(shape_x)
    n_traj = mu.shape[0]
    n_win = 1
    n_tsteps = t.shape[1]

    rel_err = np.zeros((n_traj, n_tsteps))
    nfe = np.zeros((n_traj,))

    version = get_version_str(config, f"{dim_z}_{threshold}_{degree}")
    version_dir = os.path.join(CURR_DIR, "logs_psindy", version)

    out_dir = os.path.join(CURR_DIR, "postproc_psindy", version)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(version_dir, "sindy_model.pkl"), "rb") as file:
        model = pkl.load(file)
    
    with open(os.path.join(version_dir, "modes.pkl"), "rb") as file:
        modes = pkl.load(file)
    
    #model.print()

    for i in range(n_traj):
        print(f"Trajectory {i+1}/{n_traj}")

        traj_dir = os.path.join(out_dir, f"{TRAIN_VAL_TEST}_{i}")
        pathlib.Path(traj_dir).mkdir(parents=True, exist_ok=True)

        x_i = x[i].cpu().detach().numpy()
        z_i = np.matmul(x[i].cpu().detach().flatten(1).numpy(), modes)
        t_i = t[i].cpu().detach().numpy()

        mu_i = mu[i].cpu().detach().numpy()
        f_i = f[i].cpu().detach().numpy()

        mu_expanded_i = np.tile(np.expand_dims(mu_i, 0), (n_tsteps, 1))
        mu_f_i = np.concatenate((mu_expanded_i, f_i), axis=-1)

        dz_pred = model.predict(z_i, u=mu_f_i)
        dz_i = np.diff(z_i, axis=0) / (t_i[1] - t_i[0])
        
        if dim_z <= 32:
            fig, axs = plt.subplots(dim_z, 1, figsize=(8, 4*dim_z))
            for j in range(dim_z):
                axs[j].plot(dz_i[:, j], label="True")
                axs[j].plot(dz_pred[:, j], label="Pred")
                axs[j].set_ylim([np.min(dz_i[:, j]), np.max(dz_i[:, j])])
                axs[j].legend()
            fig.show()
            fig.savefig(os.path.join(traj_dir, "dz_pred.png"))
            plt.close(fig)

        counter = {"count": 0}
        def rhs(t, x):
            counter["count"] += 1
            return model.predict(x.reshape(1, -1)).flatten()

        sol = solve_ivp(rhs, (t_i[0], t_i[-1]), z_i[0], t_eval=t_i)
        z_sim = sol.y.T
        print(f"Function evaluations: {counter['count']}")
        nfe[i] = counter['count']

        '''
        z_sim = model.simulate(
            z_i[0], #u=mu_f_i,
            t=t_i,
            integrator_kws={'atol': 1e-8, 'method': 'RK45', 'rtol': 1e-6}
        )
        '''
        z_sim_i = z_sim#np.concatenate((z_i[0:1], z_sim))
        x_sim_i = np.reshape(np.matmul(z_sim_i, modes.T), (z_sim_i.shape[0], *shape_x))

        tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]
        cmap = plt.get_cmap("ocean")

        if i == 0:
            for j_plot, j_t in enumerate(tsamples):
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(x_sim_i[j_t, 0], cmap=cmap, vmin=-1, vmax=1)
                plt.axis("off")
                fig.tight_layout()
                fig.savefig(os.path.join(compplot_out_dir, f"test_{i}_{j_plot}_pred.pdf"), format='pdf')
                fig.show()

        if dim_z <= 32:
            fig, axs = plt.subplots(dim_z, 1, figsize=(8, 4*dim_z))
            for j in range(dim_z):
                axs[j].plot(z_i[:, j], label="True")
                axs[j].plot(z_sim_i[:, j], label="Pred")
                axs[j].set_ylim([np.min(z_i[:, j]), np.max(z_i[:, j])])
                axs[j].legend()
            fig.show()
            fig.savefig(os.path.join(traj_dir, "z_sim.png"))
            plt.close(fig)

        tsamples = [0, int(n_tsteps/4), int(n_tsteps/2), int(3*n_tsteps/4), n_tsteps-1]
        n_tstep_sim = x_sim_i.shape[0]
        
        """
        fig, axs = plt.subplots(len(tsamples), 2, figsize=(8, 4*len(tsamples)))
        for j, tj in enumerate(tsamples):
            if tj >= n_tstep_sim:
                break
            axs[j, 0].imshow(x_i[tj, 0])
            axs[j, 1].imshow(x_sim_i[tj, 0])
            axs[j, 0].set_title(f"True {tj}")
            axs[j, 1].set_title(f"Pred {tj}")
        fig.show()
        fig.savefig(os.path.join(traj_dir, "x_sim.png"))
        plt.close(fig)
        """

        fig = plt.figure(figsize=(8, 4))
        resid = np.reshape(x_i[:n_tstep_sim] - x_sim_i, (n_tstep_sim, -1))
        x_i_flat = np.reshape(x_i[:n_tstep_sim], (n_tstep_sim, -1))
        rel_err[i, :n_tstep_sim] = np.linalg.norm(resid, axis=-1) / np.linalg.norm(x_i_flat, axis=-1)
        plt.plot(t_i[:n_tstep_sim], rel_err[i, :n_tstep_sim], label="Error")
        fig.show()
        fig.savefig(os.path.join(traj_dir, "error.png"))
        plt.close(fig)

        with open(os.path.join(traj_dir, "nfe.txt"), "w") as file:
            file.write(f"NFE = {nfe[i]}")

        print(f"Mean relative error: {np.mean(rel_err[i])}")

    print(f"Mean relative error: {np.mean(rel_err.flatten())}, Std Dev: {np.std(rel_err.flatten())}", flush=True)

    np.set_printoptions(threshold=np.inf)
    with open(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_error.txt"), "w") as f:
        f.write(np.array2string(rel_err, precision=5))
        f.write("\n")
        f.write(f"Mean: {np.mean(rel_err):.5f}, Std Dev: {np.std(rel_err):.5f}\n")
    
    with open(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_nfe.txt"), "w") as file:
        file.write(np.array2string(nfe))
        file.write("\n")
        file.write(f"Mean NFE per traj: {np.mean(nfe):.5f}\n")

if __name__ == "__main__":
    main()