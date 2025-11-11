import os
import pathlib
import h5py as h5
import torch
import pickle as pkl

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
EXT = ".h5"

def full_data_path(data_file: str) -> str:
    return os.path.join(CURR_DIR, data_file + EXT)

def load_data(data_file: str = "data_900_50_50_noisy") -> dict:
    print(f"Loading data from {data_file}... ", flush=True, end="")
    data_path = full_data_path(data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    with h5.File(data_path, "r") as f:
        data = {key: torch.from_numpy(f[key][()]).to(torch.float32) for key in f.keys()}
    
    print("Done.", flush=True)
    return data

if __name__ == "__main__":
    data = load_data()
    print("Data loaded successfully.")
    for key, value in data.items():
        print(f"{key}: {value.shape}")
        print(type(value))
        print(value.dtype)