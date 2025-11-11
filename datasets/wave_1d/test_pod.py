import torch
import os
import pathlib
import numpy as np
#from sklearn.utils.extmath import randomized_svd

from datasets.wave_1d.load_data import load_data

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def train_pod(data, dim_z: int, overwrite: bool = False):
    snapshots = data["train_x"].flatten(0, 1).flatten(1)
    print("Snapshots shape:", snapshots.shape)
    U, S, VH = torch.linalg.svd(snapshots.T, full_matrices=False)
    #U, S, VH = randomized_svd(snapshots.T.cpu().numpy(), n_components=dim_z, n_iter=5, random_state=42)
    #U = torch.from_numpy(U)
    print("SVD done, U shape:", U.shape, "S shape:", S.shape, "VH shape:", VH.shape)
    modes = U[:, :dim_z]
    latent_snapshots = torch.einsum("ij,jk->ik", [snapshots, modes])
    latent_trajs = latent_snapshots.reshape(data["train_x"].shape[0], data["train_x"].shape[1], dim_z)

    energy_frac = (S[:dim_z]**2).sum() / (S**2).sum()
    print(f"Energy fraction captured by top {dim_z} modes: {energy_frac:.4f}")

    return modes, latent_trajs

def main(overwrite: bool = False):
    data = load_data()
    modes, latent_trajs = train_pod(data, dim_z=25, overwrite=overwrite)

if __name__ == "__main__":
    main(overwrite=False)