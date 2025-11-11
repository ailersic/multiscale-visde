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
from experiments_refac.kdv_1d.def_model import create_latent_sde
from experiments_refac.kdv_1d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.kdv_1d.load_data import load_data

plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rc('font', family='Garamond')

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
    dim_x = 1000#int(np.prod(shape_x))
    sigma = dim_x // config.dim_z_macro

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

    #cmap = "coolwarm"
    kernel = model.encoder.macro_mean_net.net[1].weight.squeeze().detach().cpu().numpy()
    #lim = max(abs(kernel.min()), abs(kernel.max()))

    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    #print(kernel)
    ax.plot(kernel, color="black", lw=2)
    #ax.set_title("Macroscale kernel")
    
    sigmas = np.arange(0, kernel.shape[0], sigma)
    ax.set_xticks(sigmas)
    ax.set_xticklabels([f"${i-kernel.shape[0]//2}$" for i in sigmas])
    ax.set_xlabel("High-resolution mesh index")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kernel.pdf"))
    print(os.path.join(out_dir, "kernel.pdf"))
    plt.close()

if __name__ == "__main__":
    main()