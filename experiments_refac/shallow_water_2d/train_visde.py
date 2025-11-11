import torch
from torch.utils.data import DataLoader

import os
import pathlib
import shutil
from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning import loggers
#from pytorch_lightning.profilers import SimpleProfiler
#from pytorch_lightning.callbacks import EarlyStopping

import visde
from experiments_refac.shallow_water_2d.def_model import create_latent_sde, augment_latent_sde
from experiments_refac.shallow_water_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.shallow_water_2d.load_data import full_data_path

torch.manual_seed(42)
torch.backends.cudnn.benchmark=True
print(f"cuDNN: {torch.backends.cudnn.is_available()}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def get_dataloaders(data_file: str,
                    n_win: int,
                    n_batch: int
) -> tuple[DataLoader, DataLoader]:
    full_path = full_data_path(data_file)
    print(f"Using data from {full_path}", flush=True)

    train_data = visde.MultiTrajHDF5(full_path, "train", n_win)
    val_data = visde.MultiTrajHDF5(full_path, "val", n_win)

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

def main(config: CaseConfig = DEFAULT_CONFIG, overwrite: bool = True) -> None:
    print(f"CUDA: {torch.cuda.is_available()}")

    train_dataloader, val_dataloader = get_dataloaders(config.data_file, config.n_win, config.n_batch)

    version = get_version_str(config)
    print(f"Version string: {version}")

    if config.augment and config.dim_z_micro >= 1:
        old_config = deepcopy(config)
        old_config.dim_z_micro = config.dim_z_micro - 1
        if old_config.dim_z_micro == 0:
            old_config.lr = 1e-3
        model = augment_latent_sde(old_config, config, device)
    else:
        model = create_latent_sde(config, device)
    
    if os.path.exists(os.path.join(CURR_DIR, "logs_visde", version)):
        if overwrite:
            print(f"Version {version} already exists. Overwriting...", flush=True)
            while os.path.exists(os.path.join(CURR_DIR, "logs_visde", version)):
                try:
                    shutil.rmtree(os.path.join(CURR_DIR, "logs_visde", version))
                except OSError as e:
                    continue
        else:
            print(f"Version {version} already exists. Skipping...", flush=True)
            return
    
    tensorboard = loggers.TensorBoardLogger(CURR_DIR, name="logs_visde", version=version)
    #profiler = SimpleProfiler(dirpath=".", filename="perf_logs")

    trainer = pl.Trainer(
        accelerator=device.type,
        log_every_n_steps=1,
        max_epochs=config.max_epochs,
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