import torch
import torchsde

import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings

import visde
from experiments_refac.cylinder_2d.def_model import create_latent_sde
from experiments_refac.cylinder_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.cylinder_2d.load_data import load_data

sys.setrecursionlimit(2000)
plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
TRAIN_VAL_TEST = "test"  # "train", "val", or "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def plot_z_micro(traj_dir: str,
                 t: np.ndarray,
                 z_micro_enc: np.ndarray,
                 z_micro_mean: np.ndarray,
                 z_micro_std: np.ndarray
                 ) -> None:
    dim_z_micro = z_micro_enc.shape[-1]

    if dim_z_micro > 0:
        fig_zmicro, ax_zmicro = plt.subplots(figsize=(12, 6*dim_z_micro), nrows=dim_z_micro, ncols=1, squeeze=False)
        for j in range(dim_z_micro):
            ax_zmicro[j, 0].plot(t, z_micro_mean[:, j], "b-", label="Latent Dynamics")
            ax_zmicro[j, 0].fill_between(t, z_micro_mean[:, j] - z_micro_std[:, j], z_micro_mean[:, j] + z_micro_std[:, j], alpha=0.2)
            ax_zmicro[j, 0].plot(t, z_micro_enc[:, j], "r--", label="Encoded Truth")
            ax_zmicro[j, 0].legend()
            ax_zmicro[j, 0].set_xlabel("Time")
            ax_zmicro[j, 0].set_ylabel(f"Micro latent variable {j}")
        fig_zmicro.savefig(os.path.join(traj_dir, "z_micro.png"))
        fig_zmicro.show()
        plt.close(fig_zmicro)

def plot_error(traj_dir: str,
               t: np.ndarray,
               aenc_rel_err: np.ndarray,
               rel_err: np.ndarray
               ) -> None:
    figerr, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, rel_err, label="Error")
    ax.plot(t, aenc_rel_err, label="AEnc Error")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Relative Error")
    ax.set_yscale("log")
    ax.set_title(f"Mean error: {np.mean(rel_err):.4f}")
    ax.legend()
    ax.grid(True)
    figerr.savefig(os.path.join(traj_dir, "error.png"))
    figerr.show()
    plt.close(figerr)

def plot_pred_vs_true(traj_dir: str,
                      t: np.ndarray,
                      x_true: np.ndarray,
                      x_mean: np.ndarray,
                      x_std: np.ndarray,
                      x_macro: np.ndarray,
                      x_micro: np.ndarray
                      ) -> None:
    n_tsteps = t.shape[0]
    tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]

    channels = ['u', 'v']

    for i_chan in range(x_true.shape[1]):
        fig, axgrid = plt.subplots(nrows=len(tsamples), ncols=6, figsize=(32, 2*len(tsamples)), sharex=True, sharey=True, layout="constrained")
        
        state_min = 0
        state_max = 0
        error_min = 0
        error_max = 0

        for j, j_t in enumerate(tsamples):
            state_min = min(state_min, x_true[j_t, i_chan].min())
            state_max = max(state_max, x_true[j_t, i_chan].max())
            error_max = max(error_max, max(np.abs(x_true[j_t, i_chan] - x_mean[j_t, i_chan]).max(), x_std[j_t, i_chan].max()))
        
        for j, j_t in enumerate(tsamples):
            cmap = "coolwarm"
            
            axgrid[j, 0].imshow(x_true[j_t, i_chan], cmap=cmap, vmin=state_min, vmax=state_max)
            axgrid[j, 0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            
            axgrid[j, 1].imshow(x_mean[j_t, i_chan], cmap=cmap, vmin=state_min, vmax=state_max)
            axgrid[j, 1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            
            axgrid[j, 2].imshow(x_macro[j_t, i_chan], cmap=cmap, vmin=state_min, vmax=state_max)
            axgrid[j, 2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            
            if x_micro is not None:
                axgrid[j, 3].imshow(x_micro[j_t, i_chan], cmap=cmap, vmin=state_min, vmax=state_max)
            axgrid[j, 3].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

            axgrid[j, 4].imshow(np.abs(x_true[j_t, i_chan] - x_mean[j_t, i_chan]), cmap='afmhot', vmin=error_min, vmax=error_max)
            axgrid[j, 4].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

            axgrid[j, 5].imshow(x_std[j_t, i_chan], cmap='afmhot', vmin=error_min, vmax=error_max)
            axgrid[j, 5].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

            if j == 0:
                axgrid[0, 0].set_title("Truth")
                axgrid[0, 1].set_title("Mean")
                axgrid[0, 2].set_title("Macro")
                axgrid[0, 3].set_title("Micro")
                axgrid[0, 4].set_title("Error")
                axgrid[0, 5].set_title("Std. Dev.")

            axgrid[j, 0].set_ylabel(f"$t={t[j_t]:.2f}$")
    
        fig.savefig(os.path.join(traj_dir, f"pred_vs_true_{channels[i_chan]}.pdf"), format='pdf')
        fig.show()
        plt.close(fig)

def main(config: CaseConfig = DEFAULT_CONFIG) -> None:
    data = load_data(config.data_file)

    mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    n_tsteps_trunc = -1

    t = t[:, :n_tsteps_trunc]
    x = x[:, :n_tsteps_trunc, :]
    f = f[:, :n_tsteps_trunc, :]

    dim_z = config.dim_z_macro + config.dim_z_micro
    shape_x = x.shape[2:]

    n_traj = mu.shape[0]
    n_win = config.n_win
    n_batch_encoder = 64
    n_batch_decoder = 64
    n_tsteps = t.shape[1]

    rel_err = np.zeros((n_traj, n_tsteps))
    aenc_rel_err = np.zeros((n_traj, n_tsteps))

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
    
    with torch.no_grad():
        for i_traj in range(0, n_traj):
            print(f"Trajectory {i_traj+1}/{n_traj} from {TRAIN_VAL_TEST} set", flush=True)

            traj_dir = os.path.join(out_dir, f"{TRAIN_VAL_TEST}_traj_{i_traj}")
            pathlib.Path(traj_dir).mkdir(parents=True, exist_ok=True)

            print("Integrating SDE...", flush=True, end="")

            mu_i = mu[i_traj].unsqueeze(0)
            mu_i_batch = mu_i.repeat((n_batch_encoder, 1))
            t_i = t[i_traj]
            x0_i = x[i_traj, :n_win, :].unsqueeze(0)
            f_i = f[i_traj]

            z0_i = model.encoder.sample(n_batch_encoder, mu_i, x0_i)
            sde = visde.sde.SDE(model.drift, model.dispersion, mu_i, t_i, f_i)
            # suppress warnings and no gradient tracking
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z_int = torchsde.sdeint(sde, z0_i, t_i, **sde_options)
            print("done", flush=True)

            z_enc = torch.zeros(n_tsteps, 1, dim_z).to(device)
            #z_int.shape = (n_tsteps, n_batch, dim_z)
            z_mean = z_int.mean(dim=1)
            z_std = z_int.std(dim=1)

            x_true = x[i_traj]
            x_mean = torch.zeros(n_tsteps, *shape_x).to(device)
            x_std = torch.zeros(n_tsteps, *shape_x).to(device)

            x_macro_mean = torch.zeros(n_tsteps, *shape_x).to(device)
            x_micro_mean = torch.zeros(n_tsteps, *shape_x).to(device)

            print("Decoding trajectory...", flush=True, end="")
            for j_t in range(n_tsteps):
                if j_t % (n_tsteps // 5) == 0:
                    print(f"{j_t}...", flush=True, end="")

                x_macro_mean[j_t] = model.decoder.macro_mean(mu_i_batch, z_int[j_t, :, :config.dim_z_macro]).mean(dim=0)
                if config.dim_z_micro > 0:
                    x_micro_mean[j_t] = model.decoder.micro_mean(mu_i_batch, z_int[j_t, :, config.dim_z_macro:]).mean(dim=0)

                x_samples = model.decoder.sample(n_batch_decoder, mu_i_batch, z_int[j_t]).detach()
                x_mean[j_t] = x_samples.mean(dim=0)
                x_std[j_t] = x_samples.std(dim=0)

                rel_err[i_traj, j_t] = ((x_mean[j_t] - x[i_traj, j_t]).pow(2).sum() / x[i_traj, j_t].pow(2).sum()).sqrt().item()

                z_enc[j_t], _ = model.encoder(mu_i, x[i_traj, j_t:(j_t+n_win)].unsqueeze(0))
                x_rec_ij, _ = model.decoder(mu_i, z_enc[j_t])

                aenc_rel_err[i_traj, j_t] = ((x_rec_ij - x[i_traj, j_t]).pow(2).sum() / x[i_traj, j_t].pow(2).sum()).sqrt().item()
            print("done", flush=True)

            print("Plotting...", flush=True, end="")
            
            plot_z_micro(traj_dir,
                        t_i.cpu().detach().numpy(),
                        z_enc[:, 0, config.dim_z_macro:].cpu().detach().numpy(),
                        z_mean[:, config.dim_z_macro:].cpu().detach().numpy(),
                        z_std[:, config.dim_z_macro:].cpu().detach().numpy()
                        )

            plot_error(traj_dir,
                    t_i.cpu().detach().numpy(),
                    aenc_rel_err[i_traj],
                    rel_err[i_traj]
                    )
            
            plot_pred_vs_true(traj_dir,
                            t_i.cpu().detach().numpy(),
                            x_true.cpu().detach().numpy(),
                            x_mean.cpu().detach().numpy(),
                            x_std.cpu().detach().numpy(),
                            x_macro_mean.cpu().detach().numpy(),
                            x_micro_mean.cpu().detach().numpy()
                            )
            
            print("done", flush=True)

            with open(os.path.join(traj_dir, "nfe.txt"), "w") as file:
                file.write(f"NFEs = {model.drift.macro_drift_net.nfe}")

            with open(os.path.join(traj_dir, "error.txt"), "w") as file:
                np.set_printoptions(threshold=sys.maxsize)
                file.write(np.array2string(rel_err[i_traj], precision=5))
                file.write("\n")
                file.write(f"Mean: {np.mean(rel_err[i_traj]):.5f}, Std Dev: {np.std(rel_err[i_traj]):.5f}\n")

            print(f"Error Mean: {np.mean(rel_err[i_traj])}, Std Dev: {np.std(rel_err[i_traj])}", flush=True)
            print("-"*80, flush=True)

    print(f"Error Mean: {np.mean(rel_err.flatten())}, Std Dev: {np.std(rel_err.flatten())}", flush=True)

    with open(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_error.txt"), "w") as file:
        np.set_printoptions(threshold=sys.maxsize)
        file.write(np.array2string(rel_err.flatten(), precision=5))
        file.write("\n")
        file.write(f"Mean: {np.mean(rel_err.flatten()):.5f}, Std Dev: {np.std(rel_err.flatten()):.5f}\n")
    
    print("All done!", flush=True)

if __name__ == "__main__":
    main()