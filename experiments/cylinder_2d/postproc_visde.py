import torch
from torch import Tensor
import torchsde

import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

import visde
from experiments.cylinder_2d.def_model import create_latent_sde

plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data.pkl"
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(dim_z_macro: int = 2*32*8, dim_z_micro: int = 0, max_epochs: int = 1000, lr: float = 1e-3, lr_sched_freq: int = 1000):
    with open(os.path.join(CURR_DIR, DATA_FILE), "rb") as f:
        data = pkl.load(f)
    
    mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    dim_z = dim_z_macro + dim_z_micro
    dim_x = x.shape[-1]
    shape_x = x.shape[2:]

    n_traj = mu.shape[0]
    n_win = 1
    n_batch = 16
    n_batch_decoder = 16
    n_tsteps = t.shape[1]

    norm_rmse = np.zeros(n_traj)

    sde_options = {
        'method': 'srk',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-4,
        'atol': 1e-6
    }

    dummy_model = create_latent_sde(dim_z_macro, dim_z_micro, n_batch, n_win, lr, lr_sched_freq, DATA_FILE, device)
    version = "_".join([str(dim_z_macro), str(dim_z_micro), str(max_epochs), str(lr), str(lr_sched_freq)])
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

    tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]

    fig = plt.figure(figsize=(32*n_traj, 3*len(tsamples)))
    axgrid = AxesGrid(fig, 111,
                    nrows_ncols=(len(tsamples), 6*n_traj),
                    axes_pad=0.20,
                    share_all=True,
                    direction="column"
                    )
    
    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    for i_traj in range(n_traj):
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

        if dim_z_micro > 0:
            fig_zmicro, ax_zmicro = plt.subplots(figsize=(12, 6*dim_z_micro), nrows=dim_z_micro, ncols=1, squeeze=False)
            for j in range(dim_z_micro):
                z_mean = zs[:, :, dim_z_macro + j].mean(dim=1).cpu().detach().numpy()
                z_std = zs[:, :, dim_z_macro + j].std(dim=1).cpu().detach().numpy()
                ax_zmicro[j, 0].plot(t_i.cpu().detach().numpy(), z_mean)
                ax_zmicro[j, 0].fill_between(t_i.cpu().detach().numpy(),
                                        z_mean - z_std,
                                        z_mean + z_std,
                                        alpha=0.2)
                ax_zmicro[j, 0].set_title(f"Micro latent variable {j}")
            fig_zmicro.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_z_micro_traj_{i_traj}.png"))
            fig_zmicro.show()

        sqerr = np.zeros(n_tsteps)
        norm_sqerr = np.zeros(n_tsteps)
        aenc_sqerr = np.zeros(n_tsteps)
        aenc_norm_sqerr = np.zeros(n_tsteps)

        z_i = torch.zeros(n_tsteps, 1, dim_z).to(device)

        print("Decoding trajectory...", flush=True)
        for j_t in range(n_tsteps):
            if j_t % 100 == 0:
                print(f"{j_t}...", flush=True)

            xs = model.decoder.sample(n_batch_decoder, mu_i_batch, zs[j_t]).detach()
            x_mean = xs.mean(dim=0)
            x_err = x_mean - x[i_traj, j_t]

            sqerr[j_t] = x_err.pow(2).sum().item()
            norm_sqerr[j_t] = sqerr[j_t] / x[i_traj, j_t].pow(2).sum().item()

            z_i[j_t], _ = model.encoder(mu_i, x[i_traj, j_t:(j_t+n_win)].unsqueeze(0))
            x_rec_ij, _ = model.decoder(mu_i, z_i[j_t])
            aenc_err = x_rec_ij - x[i_traj, j_t]

            aenc_sqerr[j_t] = aenc_err.pow(2).sum().item()
            aenc_norm_sqerr[j_t] = aenc_sqerr[j_t] / x[i_traj, j_t].pow(2).sum().item()
        print("done", flush=True)

        norm_rmse[i_traj] = np.sqrt(np.mean(norm_sqerr))
        
        print(f"Mean Normalized RMSE: {norm_rmse[i_traj]}", flush=True)

        figrmse, ax = plt.subplots(figsize=(12, 6))
        #ax.plot(rmse, label="RMSE")
        ax.plot(np.sqrt(norm_sqerr), label="Normalized Rel. Error")
        #ax.plot(aenc_rmse, label="AEnc RMSE")
        ax.plot(np.sqrt(aenc_norm_sqerr), label="AEnc Normalized Rel. Error")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Relative Error")
        ax.set_title(f"Normalized RMSE: {norm_rmse[i_traj]:.3f}")
        ax.legend()
        figrmse.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_rmse_traj_{i_traj}.png"))
        figrmse.show()

        x_true = x[i_traj].cpu().detach().numpy()

        plot_vort = False

        state_min = 0
        state_max = 0

        for j, j_t in enumerate(tsamples):
            if plot_vort:
                vort_true = np.gradient(x_true[j_t, 1], axis=0) - np.gradient(x_true[j_t, 0], axis=1)
                state_min = min(state_min, vort_true.min())
                state_max = max(state_max, vort_true.max())
            else:
                state_min = min(state_min, x_true[j_t, 0].min())
                state_max = max(state_max, x_true[j_t, 0].max())

        error_min = 0
        error_max = state_max - state_min

        for j, j_t in enumerate(tsamples):
            xs = model.decoder.sample(n_batch_decoder, mu_i_batch, zs[j_t]).detach()
            x_macro = model.decoder.decode_mean.decode_macro(zs[j_t, :, :dim_z_macro]).mean(dim=0).unflatten(0, shape_x).cpu().detach().numpy()
            if dim_z_micro > 0:
                x_micro = model.decoder.decode_mean.decode_micro(zs[j_t, :, dim_z_macro:]).mean(dim=0).unflatten(0, shape_x).cpu().detach().numpy()

            #x_mean = xs.mean(dim=0).cpu().detach().numpy()
            #x_std = xs.std(dim=0).cpu().detach().numpy()

            if plot_vort:
                vort_s = np.gradient(xs[:, 1].cpu().detach().numpy(), axis=1) - np.gradient(xs[:, 0].cpu().detach().numpy(), axis=2)
                vort_mean = vort_s.mean(axis=0)
                vort_std = vort_s.std(axis=0)
                vort_true = np.gradient(x_true[j_t, 1], axis=0) - np.gradient(x_true[j_t, 0], axis=1)

                vort_macro = np.gradient(x_macro[1], axis=0) - np.gradient(x_macro[0], axis=1)
                if dim_z_micro > 0:
                    vort_micro = np.gradient(x_micro[1], axis=0) - np.gradient(x_micro[0], axis=1)
            else:
                vort_mean = xs.mean(dim=0).cpu().detach().numpy()[0]
                vort_std = xs.std(dim=0).cpu().detach().numpy()[0]
                vort_true = x_true[j_t].copy()[0]

                vort_macro = x_macro.copy()[0]
                if dim_z_micro > 0:
                    vort_micro = x_micro.copy()[0]

            id1 = j + 6*i_traj*len(tsamples)
            id2 = j + (6*i_traj + 1)*len(tsamples)
            id3 = j + (6*i_traj + 2)*len(tsamples)
            id4 = j + (6*i_traj + 3)*len(tsamples)
            id5 = j + (6*i_traj + 4)*len(tsamples)
            id6 = j + (6*i_traj + 5)*len(tsamples)

            cmap = "coolwarm"
            
            axgrid[id1].imshow(vort_true, cmap=cmap, vmin=state_min, vmax=state_max)
            axgrid[id1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            
            axgrid[id2].imshow(vort_mean, cmap=cmap, vmin=state_min, vmax=state_max)
            axgrid[id2].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            
            axgrid[id3].imshow(vort_macro, cmap=cmap, vmin=state_min, vmax=state_max)
            axgrid[id3].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            
            if dim_z_micro > 0:
                axgrid[id4].imshow(vort_micro, cmap=cmap, vmin=state_min, vmax=state_max)
            axgrid[id4].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

            axgrid[id5].imshow(np.abs(vort_true - vort_mean), cmap='coolwarm', vmin=error_min, vmax=error_max)
            axgrid[id5].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

            axgrid[id6].imshow(vort_std, cmap='coolwarm', vmin=error_min, vmax=error_max)
            axgrid[id6].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)

            if j == 0:
                axgrid[id1].set_title("Truth")
                axgrid[id2].set_title("Mean")
                axgrid[id3].set_title("Macro")
                axgrid[id4].set_title("Micro")
                axgrid[id5].set_title("Error")
                axgrid[id6].set_title("Std. Dev.")

    for j, j_t in enumerate(tsamples):
        axgrid[j].set_ylabel(f"$t={t[0, j_t]:.2f}$")
    
    fig.savefig(os.path.join(out_dir, f"{TRAIN_VAL_TEST}_pred_vs_true.pdf"), format='pdf')
    fig.show()

    print(f"Normalized RMSE Mean: {np.mean(norm_rmse)}, Std Dev: {np.std(norm_rmse)}", flush=True)

if __name__ == "__main__":
    main()