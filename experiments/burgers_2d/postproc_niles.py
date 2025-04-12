import torch
import torchsde
import torchdiffeq

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from experiments.burgers_2d.train_niles import niLES, dyn

plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data.pkl"
TRAIN_VAL_TEST = "test"
CLOSURE = False

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(dim_z_macro: int = 8**2, dim_z_micro: int = 0, max_epochs: int = 10, lr: float = 1e-3, lr_sched_freq: int = 100):
    with open(os.path.join(CURR_DIR, DATA_FILE), "rb") as f:
        data = pkl.load(f)
    
    mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    n_traj = mu.shape[0]
    n_tsteps = t.shape[1]

    #dummy_model = niLES(dim_z_macro, lr, lr_sched_freq)
    version = "_".join([str(dim_z_macro), str(dim_z_micro), str(max_epochs), str(lr), str(lr_sched_freq)])
    ckpt_dir = os.path.join(CURR_DIR, "logs_niles", version, "checkpoints")
    out_dir = os.path.join(CURR_DIR, "postproc_niles", version)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(ckpt_dir):
        if file.endswith(".ckpt"):
            ckpt_file = file
    
    model = niLES.load_from_checkpoint(os.path.join(ckpt_dir, ckpt_file),
                                       dim_z_macro=dim_z_macro,
                                       lr=lr,
                                       lr_sched_freq=lr_sched_freq
                                       ).to(device)
    model.eval()

    tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]

    fig, axs = plt.subplots(len(tsamples), 2*n_traj, figsize=(12*n_traj, 6*len(tsamples)))
    
    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    for i_traj in range(n_traj):
        print(f"Integrating SDE for trajectory {TRAIN_VAL_TEST} {i_traj}...", flush=True)

        t_i = t[i_traj]
        x_i = x[i_traj]
        sigma = x_i.shape[-1] // model.coarse_grid_lin

        x_i_pred = torch.zeros_like(x_i[:, :, ::sigma, ::sigma])
        x_i_true = x_i[:, :, ::sigma, ::sigma]
        x_i_pred[0] = x_i_true[0]

        for j_t in range(0, n_tsteps, model.n_batch - 1):
            t_ij = t_i[j_t:(j_t + model.n_batch)]
            x_int = torchdiffeq.odeint(dyn, x_i_pred[j_t].flatten(), t_ij, method="euler").unflatten(1, (1, model.coarse_grid_lin, model.coarse_grid_lin))
            
            if CLOSURE:
                z0_mean, z0_cov = model.encode(x_i_pred[j_t])
                z0 = torch.distributions.MultivariateNormal(z0_mean, z0_cov).sample((model.n_zsamples,))
            
                z_sde = torchsde.sdeint(model.latent_sde, z0, t_ij, dt=1e-3, adaptive=False)
                x_clo = model.dec(z_sde.flatten(0, 1)).reshape(t_ij.shape[0], -1, 1, model.coarse_grid_lin, model.coarse_grid_lin).mean(1)
            
                x_i_pred[j_t:(j_t + t_ij.shape[0])] = x_int + x_clo
            else:
                x_i_pred[j_t:(j_t + t_ij.shape[0])] = x_int
        print("done", flush=True)

        for j_t, t_j in enumerate(tsamples):
            x_ij_pred = x_i_pred[t_j]
            x_ij_true = x_i_true[t_j]

            axs[j_t, 2*i_traj].imshow(x_ij_pred[0].cpu().detach().numpy(), cmap='viridis')
            axs[j_t, 2*i_traj].set_title(f"Pred. {TRAIN_VAL_TEST} {i_traj} t={t_i[t_j]:.2f}")
            axs[j_t, 2*i_traj].axis('off')

            axs[j_t, 2*i_traj+1].imshow(x_ij_true[0].cpu().detach().numpy(), cmap='viridis')
            axs[j_t, 2*i_traj+1].set_title(f"True {TRAIN_VAL_TEST} {i_traj} t={t_i[t_j]:.2f}")
            axs[j_t, 2*i_traj+1].axis('off')
        
        fig_err, ax_err = plt.subplots(1, 1, figsize=(12, 6))
        err_i = x_i_pred.flatten(1) - x_i_true.flatten(1)
        mse_i = torch.mean(err_i**2, dim=1)
        ax_err.plot(t_i.cpu().detach().numpy(), mse_i.cpu().detach().numpy())
        ax_err.set_title(f"MSE {TRAIN_VAL_TEST} {i_traj}; Avg MSE: {mse_i.mean().cpu().detach().numpy():.4f}")
        ax_err.set_xlabel("t")
        ax_err.set_ylabel("MSE")
        fig_err.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_{CLOSURE}_mse_{i_traj}.pdf"), format='pdf')
        fig_err.show()
    
    fig.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_{CLOSURE}_pred_vs_true.pdf"), format='pdf')
    fig.show()

if __name__ == "__main__":
    main()