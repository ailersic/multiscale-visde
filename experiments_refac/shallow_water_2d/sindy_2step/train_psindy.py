import torch
import pickle as pkl
import os
import pathlib
import numpy as np
import time
from sklearn.utils.extmath import randomized_svd
from matplotlib import pyplot as plt

from experiments_refac.shallow_water_2d.sindy_2step.def_model import create_sindy_model
from experiments_refac.shallow_water_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str
from datasets.shallow_water_2d.load_data import load_data

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def train_pod(data, dim_z: int, overwrite: bool = False):
    snapshots = data["train_x"].flatten(0, 1).flatten(1)
    print("Snapshots shape:", snapshots.shape)
    #U, S, VH = torch.linalg.svd(snapshots.T, full_matrices=False)
    U, S, VH = randomized_svd(snapshots.T.cpu().numpy(), n_components=dim_z, n_iter=5, random_state=42)
    U = torch.from_numpy(U)
    print("SVD done, U shape:", U.shape, "S shape:", S.shape, "VH shape:", VH.shape)
    fig, ax = plt.subplots()
    ax.plot(S)
    ax.set_yscale("log")
    ax.set_ylabel("Sing vals")
    ax.set_xlabel("Sing val index")
    ax.grid()
    fig.show()
    fig.savefig(os.path.join(CURR_DIR, "sing_vals.png"))
    plt.close(fig)
    modes = U[:, :dim_z]
    latent_snapshots = torch.einsum("ij,jk->ik", [snapshots, modes])
    latent_trajs = latent_snapshots.reshape(data["train_x"].shape[0], data["train_x"].shape[1], dim_z)

    return modes, latent_trajs

def main(dim_z: int = 2,
        threshold: float = 1e-1,
        degree: int = 3,
        config: CaseConfig = DEFAULT_CONFIG,
        overwrite: bool = True
) -> None:
    print("CUDA:", torch.cuda.is_available())
    start_time = time.time()

    version = get_version_str(config, f"{dim_z}_{threshold}_{degree}")
    print(version)
    version_dir = os.path.join(CURR_DIR, "logs_psindy", version)

    print("Loading data...", flush=True)
    data = load_data(config.data_file)

    n_traj = data["train_mu"].shape[0]
    n_tsteps = data["train_x"].shape[1]
    n_win = 1

    dim_mu = data["train_mu"].shape[1]
    dim_f = data["train_f"].shape[1]

    mu = data["train_mu"]
    x = data["train_x"]
    f = data["train_f"]
    t = data["train_t"]
    dt = (data["train_t"][0, 1] - data["train_t"][0, 0]).item()

    # train latent map
    print("Training POD...", flush=True)
    modes, latent_trajs = train_pod(data, dim_z, overwrite)

    # train latent sindy model
    
    z_np = latent_trajs.cpu().detach().numpy()

    print("Assembling data...", flush=True)
    expanded_mu = mu.unsqueeze(1).expand(-1, n_tsteps, -1)
    u_np = torch.cat((expanded_mu, f), dim=-1).cpu().detach().numpy()

    z = [zi.copy() for zi in z_np]
    u = [ui.copy() for ui in u_np]
    dz = [np.diff(zi, axis=0)/dt for zi in z]

    z = [zi[:-1] for zi in z]
    u = [ui[:-1] for ui in u]

    print("Going to train SINDy model...", flush=True)
    model = create_sindy_model(z, dz, u, dt, threshold, degree)
    model.print()
    print(model.score(z, u=u, t=dt, x_dot=dz, multiple_trajectories=True))

    pathlib.Path(version_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(version_dir, "sindy_model.pkl"), "wb") as f:
        pkl.dump(model, f)
        
    with open(os.path.join(version_dir, "modes.pkl"), "wb") as f:
        pkl.dump(modes.cpu().detach().numpy(), f)
    
    with open(os.path.join(version_dir, "train_time.txt"), "w") as f:
        f.write(f"{time.time() - start_time:.2f} seconds")
    print("Training time:", time.time() - start_time)

if __name__ == "__main__":
    main()