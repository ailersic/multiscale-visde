import torch
from torch.utils.data import DataLoader

import pickle as pkl
import os
import pathlib
import shutil
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers
#from pytorch_lightning.profilers import SimpleProfiler
#from pytorch_lightning.callbacks import EarlyStopping

import visde
from experiments.burgers_2d.def_model import create_latent_sde

torch.manual_seed(42)
torch.backends.cudnn.benchmark=True
print(f"cuDNN: {torch.backends.cudnn.is_available()}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
DATA_FILE = "data.pkl"

def get_dataloaders(n_win: int,
                    n_batch: int
) -> tuple[DataLoader, DataLoader]:
    with open(os.path.join(CURR_DIR, DATA_FILE), "rb") as f:
        data = pkl.load(f)

    train_data = visde.MultiEvenlySpacedTensors(data["train_mu"], data["train_t"], data["train_x"], data["train_f"], n_win)
    val_data = visde.MultiEvenlySpacedTensors(data["val_mu"], data["val_t"], data["val_x"], data["val_f"], n_win)

    train_sampler = visde.MultiTemporalSampler(train_data, n_batch, n_repeats=1)
    train_dataloader = DataLoader(
        train_data,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=train_sampler,
        pin_memory=True
    )
    val_sampler = visde.MultiTemporalSampler(val_data, n_batch, n_repeats=1)
    val_dataloader = DataLoader(
        val_data,
        num_workers=47,
        persistent_workers=True,
        batch_sampler=val_sampler,
        pin_memory=True
    )

    return train_dataloader, val_dataloader

def main(overwrite: bool = False, dim_z_macro: int = 64, dim_z_micro: int = 5, max_epochs: int = 500, lr: float = 5e-4, lr_sched_freq: int = np.inf):
    n_win = 1
    n_batch = 128
    print(f"CUDA: {torch.cuda.is_available()}")

    train_dataloader, val_dataloader = get_dataloaders(n_win, n_batch)
    model = create_latent_sde(dim_z_macro, dim_z_micro, n_batch, n_win, lr, lr_sched_freq, DATA_FILE, device)

    version = "_".join([str(dim_z_macro), str(dim_z_micro), str(max_epochs), str(lr), str(lr_sched_freq)])
    if os.path.exists(os.path.join(CURR_DIR, "logs_visde", version)):
        if overwrite:
            print(f"Version {version} already exists. Overwriting...", flush=True)
            shutil.rmtree(os.path.join(CURR_DIR, "logs_visde", version))
        else:
            print(f"Version {version} already exists. Skipping...", flush=True)
            return
    
    tensorboard = loggers.TensorBoardLogger(CURR_DIR, name="logs_visde", version=version)
    #profiler = SimpleProfiler(dirpath=".", filename="perf_logs")

    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=max_epochs,
        logger=tensorboard,
        check_val_every_n_epoch=5,
        #profiler=profiler,
        #callbacks=[EarlyStopping(monitor="val/norm_rmse", mode="min")]
    )
    # ---------------------- training ---------------------- #
    trainer.fit(model, train_dataloader, val_dataloader)
    #print(profiler.summary())

if __name__ == "__main__":
    main()