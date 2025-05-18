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
from experiments.burgers_2d.def_model import create_latent_sde

plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data_20_5_5.pkl"
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def plot_micro_state(traj_dir, t_i, z_mean, z_std):
    dim_z_micro = z_mean.shape[1]
    fig_zmicro, ax_zmicro = plt.subplots(figsize=(12, 6*dim_z_micro), nrows=dim_z_micro, ncols=1, squeeze=False)
    for j in range(dim_z_micro):
        ax_zmicro[j, 0].plot(t_i, z_mean[:, j])
        ax_zmicro[j, 0].fill_between(t_i,
                                    z_mean[:, j] - z_std[:, j],
                                    z_mean[:, j] + z_std[:, j],
                                    alpha=0.2)
        ax_zmicro[j, 0].set_title(f"Micro latent variable {j}")
    fig_zmicro.savefig(os.path.join(traj_dir, "z_micro_traj.png"))
    fig_zmicro.show()

def plot_error(traj_dir, t_i, norm_err, aenc_norm_err):
    figrmse, ax = plt.subplots(figsize=(12, 6))
    #ax.plot(rmse, label="RMSE")
    ax.plot(norm_err, label="Error")
    #ax.plot(aenc_rmse, label="AEnc RMSE")
    ax.plot(aenc_norm_err, label="AEnc Error")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Relative Error")
    ax.set_title(f"Mean error: {np.mean(norm_err):.4f}")
    ax.legend()
    figrmse.savefig(os.path.join(traj_dir, "error.png"))
    figrmse.show()

def plot_traj(traj_dir, t_i, x_macro, x_micro, x_mean, x_std, x_true):
    n_tsteps = t_i.shape[0]
    tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]

    fig = plt.figure(figsize=(12, 3*len(tsamples)))
    axgrid = AxesGrid(fig, 111,
                    nrows_ncols=(len(tsamples), 6),
                    axes_pad=0.20,
                    share_all=True,
                    direction="column"
                    )

    #cmap = "gist_ncar"
    cmap = "nipy_spectral"

    titles = ["Truth", "Mean", "Macro", "Micro", "Error", "Std. Dev."]

    for i in range(6):
        axgrid[i*len(tsamples)].set_title(titles[i])

    state_max = 0
    state_min = 0
    error_max = 0
    error_min = 0

    for j, j_t in enumerate(tsamples):
        state_max = max(state_max, np.max(np.abs(x_true[j_t, 0])))
        state_min = min(state_min, np.min(np.abs(x_true[j_t, 0])))

        error_max = max(error_max, np.max(x_std[0]))
    
    state_min = state_max - (state_max - state_min)*1.1

    for j, j_t in enumerate(tsamples):
        axgrid[j].imshow(x_true[j_t, 0], cmap=cmap, vmin=state_min, vmax=state_max)
        axgrid[j + len(tsamples)].imshow(x_mean[j_t, 0], cmap=cmap, vmin=state_min, vmax=state_max)
        axgrid[j + 2*len(tsamples)].imshow(x_macro[j_t, 0], cmap=cmap, vmin=state_min, vmax=state_max)
        if x_micro is not None:
            axgrid[j + 3*len(tsamples)].imshow(x_micro[j_t, 0], cmap=cmap, vmin=state_min, vmax=state_max)
        axgrid[j + 4*len(tsamples)].imshow(np.abs(x_true[j_t, 0] - x_mean[j_t, 0]), cmap='coolwarm', vmin=error_min, vmax=error_max)
        axgrid[j + 5*len(tsamples)].imshow(x_std[j_t, 0], cmap='coolwarm', vmin=error_min, vmax=error_max)

        for i in range(6):
            axgrid[j + i*len(tsamples)].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    

    for j, j_t in enumerate(tsamples):
        axgrid[j].set_ylabel(f"$t={t_i[j_t]:.2f}$")
    
    fig.savefig(os.path.join(traj_dir, "pred_vs_true.pdf"), format='pdf')
    fig.show()

def plot_snapshots(traj_dir, t_plot, x_full, x_smooth, z_macro):
    cmap = "nipy_spectral"

    full_img = x_full
    smooth_img = x_smooth
    macro_img = z_macro

    full_max = np.max(full_img)
    full_min = np.min(full_img)
    smooth_max = np.max(smooth_img)
    smooth_min = np.min(smooth_img)

    resid_img = full_img - smooth_img*(full_max - full_min)/(smooth_max - smooth_min)

    all_max = max(full_max, np.max(resid_img.max()))
    all_min = min(full_min, np.min(resid_img.min()))

    figmacro, axmacro = plt.subplots(figsize=(6, 6))
    axmacro.imshow(full_img, cmap=cmap, vmin=all_min, vmax=all_max)
    axmacro.axis('off')
    figmacro.savefig(os.path.join(traj_dir, f"full_snap_{t_plot}.pdf"), bbox_inches='tight')
    plt.close()

    figmacro, axmacro = plt.subplots(figsize=(6, 6))
    axmacro.imshow(smooth_img*(full_max - full_min)/(smooth_max - smooth_min), cmap=cmap, vmin=all_min, vmax=all_max)
    axmacro.axis('off')
    figmacro.savefig(os.path.join(traj_dir, f"smooth_snap_{t_plot}.pdf"), bbox_inches='tight')
    plt.close()

    figmacro, axmacro = plt.subplots(figsize=(6, 6))
    axmacro.imshow(resid_img, cmap=cmap, vmin=all_min, vmax=all_max)
    axmacro.axis('off')
    figmacro.savefig(os.path.join(traj_dir, f"resid_snap_{t_plot}.pdf"), bbox_inches='tight')
    plt.close()

    figmacro, axmacro = plt.subplots(figsize=(6, 6))
    axmacro.imshow(macro_img*(full_max - full_min)/(smooth_max - smooth_min), cmap=cmap, vmin=all_min, vmax=all_max)
    axmacro.axis('off')
    figmacro.savefig(os.path.join(traj_dir, f"macro_snap_{t_plot}.pdf"), bbox_inches='tight')
    plt.close()

def main(dim_z_macro: int = 64,
         dim_z_micro: int = 2,
         max_epochs: int = 100,
         lr: float = 1e-4,
         lr_sched_freq: int = 2000,
         augment: bool = True,
) -> None:
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
    n_batch = 64
    n_batch_decoder = 64
    n_tsteps = t.shape[1]

    norm_err = np.zeros((n_traj, n_tsteps))

    sde_options = {
        'method': 'euler',
        'dt': 1e-2,
        'adaptive': True,
        'rtol': 1e-3,
        'atol': 1e-5
    }

    dummy_model = create_latent_sde(dim_z_macro, dim_z_micro, n_batch, n_win, lr, lr_sched_freq, DATA_FILE, device)
    if augment and dim_z_micro > 0:
        version = "_".join([str(dim_z_macro), str(dim_z_micro), str(max_epochs), str(lr), str(lr_sched_freq), "augment"])
    else:
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
    
    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # zs will have shape (t_size, batch_size, dim_z)
    for i_traj in range(n_traj):
        print(f"Integrating SDE for trajectory {TRAIN_VAL_TEST} {i_traj}...", flush=True)
        traj_dir = os.path.join(out_dir, f"traj_{TRAIN_VAL_TEST}_{i_traj}")
        pathlib.Path(traj_dir).mkdir(parents=True, exist_ok=True)

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

        aenc_norm_err = np.zeros(n_tsteps)

        zenc = torch.zeros(n_tsteps, dim_z).to(device)
        x_macro_mean = torch.zeros(n_tsteps, *shape_x).to(device)
        x_micro_mean = torch.zeros(n_tsteps, *shape_x).to(device)
        x_mean = torch.zeros(n_tsteps, *shape_x).to(device)
        x_std = torch.zeros(n_tsteps, *shape_x).to(device)
        x_err = torch.zeros(n_tsteps, *shape_x).to(device)

        if dim_z_micro > 0:
            z_micro_mean = zs[:, :, dim_z_macro:].mean(dim=1)
            z_micro_std = zs[:, :, dim_z_macro:].std(dim=1)
        else:
            z_micro_mean = None
            z_micro_std = None

        print("Decoding trajectory...", flush=True, end="")
        for j_t in range(n_tsteps):
            if j_t % 100 == 0:
                print(f"{j_t}...", flush=True, end="")

            xs = model.decoder.sample(n_batch_decoder, mu_i_batch, zs[j_t]).detach()

            x_macro_mean[j_t] = model.decoder.decode_mean.decode_macro(zs[j_t, :, :dim_z_macro]).mean(dim=0).unflatten(0, shape_x).detach()
            if dim_z_micro > 0:
                x_micro_mean[j_t] = model.decoder.decode_mean.decode_micro(zs[j_t, :, dim_z_macro:]).mean(dim=0).unflatten(0, shape_x).detach()

            x_mean[j_t] = xs.mean(dim=0)
            x_std[j_t] = xs.std(dim=0)
            x_err[j_t] = x_mean[j_t] - x[i_traj, j_t]

            norm_err[i_traj, j_t] = np.sqrt(x_err[j_t].pow(2).sum().item() / x[i_traj, j_t].pow(2).sum().item())

            zenc[j_t], _ = model.encoder(mu_i, x[i_traj, j_t:(j_t+n_win)].unsqueeze(0))
            x_rec_ij, _ = model.decoder(mu_i, zenc[j_t].unsqueeze(0))
            aenc_err = x_rec_ij - x[i_traj, j_t]

            aenc_norm_err[j_t] = np.sqrt(aenc_err.pow(2).sum().item() / x[i_traj, j_t].pow(2).sum().item())
        print("done", flush=True)
        
        print(f"Mean error for trajectory {i_traj}: {np.mean(norm_err[i_traj])}", flush=True)

        ## Plotting

        if z_micro_mean is not None:
            plot_micro_state(traj_dir,
                             t_i.cpu().detach().numpy(),
                             z_micro_mean.cpu().detach().numpy(),
                             z_micro_std.cpu().detach().numpy()
                             )

        plot_error(traj_dir,
                   t_i.cpu().detach().numpy(),
                   norm_err[i_traj],
                   aenc_norm_err
                   )

        plot_traj(traj_dir,
                  t_i.cpu().detach().numpy(),
                  x_macro_mean.cpu().detach().numpy(),
                  x_micro_mean.cpu().detach().numpy(),
                  x_mean.cpu().detach().numpy(),
                  x_std.cpu().detach().numpy(),
                  x[i_traj].cpu().detach().numpy()
                  )
    
        #full_img = x_i[t_plot, 0].cpu().detach().numpy()
        #smooth_img = model.encoder.encode_mean.smooth_net(x_i[t_plot].unsqueeze(0).flatten(1)).reshape(128, 128).cpu().detach().numpy()
        #macro_img = model.encoder.encode_mean.macro_net(x_i[t_plot].unsqueeze(0).flatten(1)).reshape(8, 8).cpu().detach().numpy()
        t_plot = n_tsteps//2
        
        plot_snapshots(traj_dir,
                       t_plot,
                       x[i_traj, t_plot, 0].cpu().detach().numpy(),
                       model.encoder.encode_mean.smooth_net(x[i_traj, t_plot].unsqueeze(0).flatten(1)).reshape(128, 128).cpu().detach().numpy(),
                       model.encoder.encode_mean.macro_net(x[i_traj, t_plot].unsqueeze(0).flatten(1)).reshape(8, 8).cpu().detach().numpy()
                       )

    print(f"Error Mean: {np.mean(norm_err.flatten())}, Std Dev: {np.std(norm_err.flatten())}", flush=True)

    np.set_printoptions(threshold=np.inf)
    with open(os.path.join(out_dir, "error.txt"), "w") as f:
        f.write(np.array2string(norm_err, precision=5))
        f.write("\n")
        f.write(f"Mean: {np.mean(norm_err.flatten()):.5f}, Std Dev: {np.std(norm_err.flatten()):.5f}\n")
    
    print("All done!", flush=True)

if __name__ == "__main__":
    main()