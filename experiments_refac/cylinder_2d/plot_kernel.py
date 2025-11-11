import torch

import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np

import visde
from experiments_refac.cylinder_2d.def_model import create_latent_sde
from experiments_refac.cylinder_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str

plt.rcParams.update({'font.size': 12})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
TRAIN_VAL_TEST = "test"

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def main(config: CaseConfig = DEFAULT_CONFIG) -> None:
    #data = load_data(config.data_file)
    
    #mu = data[f"{TRAIN_VAL_TEST}_mu"].to(device)
    #t = data[f"{TRAIN_VAL_TEST}_t"].to(device)
    #x = data[f"{TRAIN_VAL_TEST}_x"].to(device)
    #f = data[f"{TRAIN_VAL_TEST}_f"].to(device)

    #shape_x = x.shape[2:]
    dim_x = 2*320*80#int(np.prod(shape_x))

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

    cmap = "coolwarm"
    enc_kernel = model.encoder.macro_mean_net.net[1].weight.squeeze().detach().cpu().numpy()
    dec_kernel = model.decoder.macro_mean_net.net[2].weight.squeeze().detach().cpu().numpy()
    if enc_kernel.ndim == 2:
        enc_kernel = enc_kernel[None, :, :]
        dec_kernel = dec_kernel[None, :, :]
    enc_lim = max(abs(enc_kernel.min()), abs(enc_kernel.max()))
    dec_lim = max(abs(dec_kernel.min()), abs(dec_kernel.max()))

    n_chan = enc_kernel.shape[0]
    sigma = int(np.sqrt(dim_x // config.dim_z_macro))
    print(f"Sigma: {sigma}, Kernel shape: {enc_kernel.shape}")
    sigmas_x = np.arange(0, enc_kernel.shape[1], sigma)
    sigmas_y = np.arange(0, enc_kernel.shape[2], sigma)

    fig, ax = plt.subplots(2, n_chan, figsize=(4*n_chan, 8), squeeze=False)

    for i in range(n_chan):
        im0 = ax[0, i].imshow(enc_kernel[i], aspect="equal", cmap=cmap, vmin=-enc_lim, vmax=enc_lim)
        ax[0, i].set_title(f"Encoder kernel {i}")
        
        ax[0, i].set_xticks(sigmas_x)
        ax[0, i].set_xticklabels([f"${i-enc_kernel.shape[1]//2}$" for i in sigmas_x])
        ax[0, i].set_yticks(sigmas_y)
        ax[0, i].set_yticklabels([f"${i-enc_kernel.shape[2]//2}$" for i in sigmas_y])
        plt.colorbar(im0, ax=ax[0, i], orientation="horizontal")

        im1 = ax[1, i].imshow(dec_kernel[i], aspect="equal", cmap=cmap, vmin=-dec_lim, vmax=dec_lim)
        ax[1, i].set_title(f"Decoder kernel {i}")

        ax[1, i].set_xticks(sigmas_x)
        ax[1, i].set_xticklabels([f"${i-dec_kernel.shape[1]//2}$" for i in sigmas_x])
        ax[1, i].set_yticks(sigmas_y)
        ax[1, i].set_yticklabels([f"${i-dec_kernel.shape[2]//2}$" for i in sigmas_y])
        plt.colorbar(im1, ax=ax[1, i], orientation="horizontal")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kernel.pdf"))
    plt.close()

if __name__ == "__main__":
    main()