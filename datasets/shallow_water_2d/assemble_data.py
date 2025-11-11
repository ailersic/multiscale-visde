import os
import pathlib
import h5py as h5
import torch
import matplotlib.pyplot as plt
import numpy as np
from load_data import full_data_path

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
EXT = ".h5"

def assemble_data(data_file: str = "133021"):
    print(f"Loading data from {data_file}... ", flush=True, end="")
    data_path = full_data_path(data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    with h5.File(data_path, "r") as f:
        n_traj = len(f.keys())
        n_train = int(0.90 * n_traj)
        n_val = int(0.05 * n_traj)
        n_test = n_traj - n_train - n_val

        fshape = f['0000']['data'].shape
        traj_shape = [fshape[0], fshape[3], fshape[1], fshape[2]]
        data = {}
        for split, (start, end) in zip(["train", "val", "test"],
                                       [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, n_traj)]):
            split_data = torch.zeros(end - start, *traj_shape)
            for i in range(start, end):
                key = f"{i:04d}"
                traj_data = torch.from_numpy(f[key]['data'][()]).to(torch.float32) - 1.0  # Center around 0
                split_data[i - start] = traj_data.permute(0, 3, 1, 2)  # (T, C, H, W)
            
            if end - start == 0:
                data[f"{split}_x"] = split_data
                data[f"{split}_t"] = torch.linspace(0, 1, traj_shape[0], dtype=torch.float32).unsqueeze(0).repeat(end - start, 1)
                data[f"{split}_mu"] = torch.zeros(end - start, 1, dtype=torch.float32)
                data[f"{split}_f"] = torch.zeros(end - start, traj_shape[0], 1, dtype=torch.float32)
            else:
                data[f"{split}_x"] = split_data + torch.randn_like(split_data) * 1e-2  # Add small noise
                data[f"{split}_t"] = torch.linspace(0, 1, traj_data.shape[0], dtype=torch.float32).unsqueeze(0).repeat(end - start, 1)
                data[f"{split}_mu"] = torch.zeros(end - start, 1, dtype=torch.float32)
                data[f"{split}_f"] = torch.zeros(end - start, traj_data.shape[0], 1, dtype=torch.float32)

    fig, axs = plt.subplots(ncols=2, nrows=1)
    axs[0].imshow(data["test_x"][0, 0, 0])
    axs[1].imshow(data["test_x"][0, -1, 0])
    fig.show()
    plt.savefig(os.path.join(CURR_DIR, "sample_traj.png"))

    data_file_name = f"data_{n_train}_{n_val}_{n_test}_noisy.h5"
    with h5.File(os.path.join(CURR_DIR, data_file_name), "w") as file:
        for key, value in data.items():
            file.create_dataset(key, data=value)

    print("Done.", flush=True)

if __name__ == "__main__":
    assemble_data()