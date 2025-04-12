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

plt.rcParams.update({'font.size': 12})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data.pkl"
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(dim_z_macro: int = 2*32*8, dim_z_micro: int = 5, max_epochs: int = 1000, lr: float = 1e-3, lr_sched_freq: int = np.inf):
    with open(os.path.join(CURR_DIR, DATA_FILE), "rb") as f:
        data = pkl.load(f)
    
    mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    dim_z = dim_z_macro + dim_z_micro
    shape_x = x.shape[2:]
    dim_x = int(np.prod(shape_x))

    n_traj = mu.shape[0]
    n_win = 1
    n_batch = 64
    n_batch_decoder = 64
    n_tsteps = t.shape[1]

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

    cmap = "coolwarm"
    kernel = model.encoder.encode_mean.macro_net[1].weight.squeeze().detach().cpu().numpy()
    lim = max(abs(kernel.min()), abs(kernel.max()))

    n_chan = kernel.shape[0]
    sigma = int(np.sqrt(dim_x // dim_z_macro))
    print(f"Sigma: {sigma}, Kernel shape: {kernel.shape}")
    sigmas_x = np.arange(0, kernel.shape[1], sigma)
    sigmas_y = np.arange(0, kernel.shape[2], sigma)

    fig, ax = plt.subplots(1, n_chan, figsize=(4*n_chan, 4))
    print(model.encoder.encode_mean.macro_net[1].weight)

    for i in range(n_chan):
        im0 = ax[i].imshow(kernel[i], aspect="equal", cmap=cmap, vmin=-lim, vmax=lim)
        ax[i].set_title(f"Macroscale kernel {i}")
        
        ax[i].set_xticks(sigmas_x)
        ax[i].set_xticklabels([f"${i-kernel.shape[1]//2}$" for i in sigmas_x])
        ax[i].set_yticks(sigmas_y)
        ax[i].set_yticklabels([f"${i-kernel.shape[2]//2}$" for i in sigmas_y])
        plt.colorbar(im0, ax=ax[i], orientation="horizontal")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kernel.pdf"))
    plt.close()

if __name__ == "__main__":
    main()