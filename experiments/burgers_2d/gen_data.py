from calendar import c
from sympy import Float
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from jaxtyping import Float
import pickle as pkl
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np

import visde

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def dyn(mu: Float[Tensor, "n_traj dim_mu"],
      t: Float[Tensor, "n_traj"],
      x: Float[Tensor, "n_traj dim_x dim_x"]
) -> Float[Tensor, "n_traj dim_x dim_x"]:
    visc = mu[:, 0]
    dim_x = x.shape[-1]
    h = 1.0/(dim_x - 1)
    dxdt = torch.zeros_like(x)

    dx11 = (x[:, 2:dim_x, 1:dim_x-1] - x[:, 0:dim_x-2, 1:dim_x-1])/(2*h)
    dx12 = (x[:, 1:dim_x-1, 2:dim_x] - x[:, 1:dim_x-1, 0:dim_x-2])/(2*h)
    dx21 = (x[:, 2:dim_x, 1:dim_x-1] + x[:, 0:dim_x-2, 1:dim_x-1] - 2*x[:, 1:dim_x-1, 1:dim_x-1])/(h**2)
    dx22 = (x[:, 1:dim_x-1, 2:dim_x] + x[:, 1:dim_x-1, 0:dim_x-2] - 2*x[:, 1:dim_x-1, 1:dim_x-1])/(h**2)
    dxdt[:, 1:dim_x-1, 1:dim_x-1] = visc.unsqueeze(1).unsqueeze(2)*(dx21 + dx22) - x[:, 1:dim_x-1, 1:dim_x-1]*(dx11 + dx12)

    return dxdt

def create_dataset(mu: Float[Tensor, "n_traj dim_mu"],
                   T: float,
                   n_tstep: int,
                   dim_x: int
) -> tuple[Float[Tensor, "n_traj n_tstep"],
           Float[Tensor, "n_traj n_tstep dim_x"],
           Float[Tensor, "n_traj n_tstep dim_f"]
]:
    n_traj = mu.shape[0]
    t = torch.linspace(0.0, T, n_tstep).unsqueeze(0).repeat(n_traj, 1)
    f = torch.zeros(n_traj, n_tstep, 1)

    x = torch.zeros(n_traj, n_tstep, dim_x, dim_x)
    xlin = torch.linspace(0.0, 1.0, dim_x).unsqueeze(0).unsqueeze(2).repeat(n_traj, 1, dim_x)
    ylin = torch.linspace(0.0, 1.0, dim_x).unsqueeze(0).unsqueeze(1).repeat(n_traj, dim_x, 1)

    c1 = 1.0 + 0.1*torch.randn(n_traj)
    c1 = c1.unsqueeze(1).unsqueeze(2).repeat(1, dim_x, dim_x)

    c2 = (0.2 + 0.01*torch.randn(n_traj))**2
    c2 = c2.unsqueeze(1).unsqueeze(2).repeat(1, dim_x, dim_x)

    x[:, 0, :, :] = c1*torch.exp(-(xlin - 0.3).pow(2).div(c2))*torch.exp(-(ylin - 0.3).pow(2).div(c2))

    for i in range(1, n_tstep):
        dt = (t[:, i] - t[:, i - 1]).unsqueeze(1).unsqueeze(2)
        xi1 = x[:, i - 1] + dyn(mu, t[:, i - 1], x[:, i - 1]) * dt
        xi2 = x[:, i - 1] + dyn(mu, t[:, i - 1], xi1) * dt
        x[:, i] = 0.5 * (xi1 + xi2)
    
    return t, x, f

def main():
    n_traj = 18
    n_tstep = 1001
    dim_x = 128

    train_T = 1.0
    train_mu = torch.cat([torch.linspace(0.005, 0.005, n_traj).unsqueeze(1),], dim=1)

    val_T = 1.0
    val_mu = torch.cat([torch.full([1, 1], 0.005)], dim=1)
    
    test_T = 1.0
    test_mu = torch.cat([torch.full([1, 1], 0.005)], dim=1)

    train_t, train_x, train_f = create_dataset(train_mu, train_T, n_tstep, dim_x)
    val_t, val_x, val_f = create_dataset(val_mu, val_T, n_tstep, dim_x)
    test_t, test_x, test_f = create_dataset(test_mu, test_T, n_tstep, dim_x)

    train_x += 0.001*torch.randn_like(train_x)
    val_x += 0.001*torch.randn_like(val_x)
    test_x += 0.001*torch.randn_like(test_x)

    train_x = train_x.unsqueeze(2)
    val_x = val_x.unsqueeze(2)
    test_x = test_x.unsqueeze(2)

    assert train_mu.shape == (n_traj, 1)
    assert train_t.shape == (n_traj, n_tstep)
    assert train_x.shape == (n_traj, n_tstep, 1, dim_x, dim_x)
    assert train_f.shape == (n_traj, n_tstep, 1)

    assert val_mu.shape == (1, 1)
    assert val_t.shape == (1, n_tstep)
    assert val_x.shape == (1, n_tstep, 1, dim_x, dim_x)
    assert val_f.shape == (1, n_tstep, 1)
    
    assert test_mu.shape == (1, 1)
    assert test_t.shape == (1, n_tstep)
    assert test_x.shape == (1, n_tstep, 1, dim_x, dim_x)
    assert test_f.shape == (1, n_tstep, 1)

    fig, axs = plt.subplots(3, 3)
    axs[0,0].imshow(train_x[0, 0, 0], cmap="hot", vmin=0, vmax=1)
    axs[0,1].imshow(train_x[0, n_tstep//2, 0], cmap="hot", vmin=0, vmax=1)
    axs[0,2].imshow(train_x[0, -1, 0], cmap="hot", vmin=0, vmax=1)

    axs[1,0].imshow(train_x[-1, 0, 0], cmap="hot", vmin=0, vmax=1)
    axs[1,1].imshow(train_x[-1, n_tstep//2, 0], cmap="hot", vmin=0, vmax=1)
    axs[1,2].imshow(train_x[-1, -1, 0], cmap="hot", vmin=0, vmax=1)

    axs[2,0].imshow(val_x[0, 0, 0], cmap="hot", vmin=0, vmax=1)
    axs[2,1].imshow(val_x[0, n_tstep//2, 0], cmap="hot", vmin=0, vmax=1)
    axs[2,2].imshow(val_x[0, -1, 0], cmap="hot", vmin=0, vmax=1)
    plt.show()
    plt.savefig(os.path.join(CURR_DIR, "data.png"), dpi=300)
    plt.close()

    data = {"train_mu": train_mu,
            "train_t": train_t,
            "train_x": train_x,
            "train_f": train_f,
            "val_mu": val_mu,
            "val_t": val_t,
            "val_x": val_x,
            "val_f": val_f,
            "test_mu": test_mu,
            "test_t": test_t,
            "test_x": test_x,
            "test_f": test_f}

    with open(os.path.join(CURR_DIR, "data.pkl"), "wb") as f:
        pkl.dump(data, f)

if __name__ == "__main__":
    main()