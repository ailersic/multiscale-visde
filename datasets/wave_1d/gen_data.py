from jaxtyping import Float
import torch
from torch import Tensor
import pickle as pkl
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
nfe = 0

def create_dataset(mu: Float[Tensor, "n_traj dim_mu"],
                   T: float,
                   n_tstep: int,
                   dim_x: int
) -> tuple[Float[Tensor, "n_traj n_tstep"],
           Float[Tensor, "n_traj n_tstep dim_x"],
           Float[Tensor, "n_traj n_tstep dim_x"]
]:
    n_traj = mu.shape[0]
    t = torch.linspace(0.0, T, n_tstep).unsqueeze(0).repeat(n_traj, 1)
    x_domain = torch.linspace(0, 1, dim_x, dtype=torch.float64)
    dim_f = 1

    x = torch.zeros(n_traj, n_tstep, dim_x)
    f = torch.zeros(n_traj, n_tstep, dim_f)
    for i in range(n_traj):
        width = 0.01 + 0.01 * torch.rand(1)
        speed = 1#0.8 + 0.2 * torch.rand(1)

        for j in range(0, n_tstep):
            #x[i, j, :] = ((x_domain - speed * t[i, j]) % 1.0 < width) & ((x_domain - speed * t[i, j]) % 1.0 > 0)

            dist = (x_domain - speed * t[i, j]) % 1.0
            dist = torch.minimum(dist, 1.0 - dist)
            x[i, j, :] = torch.exp(-(dist / width) ** 2)

        plt.figure()
        plt.plot(x_domain, x[i, 0, :dim_x].numpy(), label="t=0")
        plt.plot(x_domain, x[i, n_tstep//4, :dim_x].numpy(), label="t=T/4")
        plt.plot(x_domain, x[i, n_tstep//2, :dim_x].numpy(), label="t=T/2")
        plt.plot(x_domain, x[i, n_tstep*3//4, :dim_x].numpy(), label="t=3T/4")
        plt.plot(x_domain, x[i, -1, :dim_x].numpy(), label="t=T")
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(CURR_DIR, f"x{i}.png"))
        plt.close()

    return t, x, f

def main():
    n_traj_train = 20
    n_traj_val = 5
    n_traj_test = 5

    n_traj = n_traj_train + n_traj_val + n_traj_test
    n_tstep = 1001

    dim_x = 1000

    T = 1.0
    mu = torch.zeros(n_traj, 1)

    t, x, f = create_dataset(mu, T, n_tstep, dim_x)
    x = x.unsqueeze(2)  # Add a channel dimension
    x += 0.001 * torch.randn_like(x)

    train_mu, train_t, train_x, train_f = mu[:n_traj_train], t[:n_traj_train], x[:n_traj_train], f[:n_traj_train]
    val_mu, val_t, val_x, val_f = mu[n_traj_train:n_traj_train + n_traj_val], t[n_traj_train:n_traj_train + n_traj_val], x[n_traj_train:n_traj_train + n_traj_val], f[n_traj_train:n_traj_train + n_traj_val]
    test_mu, test_t, test_x, test_f = mu[n_traj_train + n_traj_val:], t[n_traj_train + n_traj_val:], x[n_traj_train + n_traj_val:], f[n_traj_train + n_traj_val:]

    assert train_mu.shape == (n_traj_train, 1)
    assert train_t.shape == (n_traj_train, n_tstep)
    assert train_x.shape == (n_traj_train, n_tstep, 1, dim_x)
    assert train_f.shape == (n_traj_train, n_tstep, 1)

    assert val_mu.shape == (n_traj_val, 1)
    assert val_t.shape == (n_traj_val, n_tstep)
    assert val_x.shape == (n_traj_val, n_tstep, 1, dim_x)
    assert val_f.shape == (n_traj_val, n_tstep, 1)

    assert test_mu.shape == (n_traj_test, 1)
    assert test_t.shape == (n_traj_test, n_tstep)
    assert test_x.shape == (n_traj_test, n_tstep, 1, dim_x)
    assert test_f.shape == (n_traj_test, n_tstep, 1)

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

    with open(os.path.join(CURR_DIR, f"data_gauss_{n_traj_train}_{n_traj_val}_{n_traj_test}.pkl"), "wb") as f:
        pkl.dump(data, f)

if __name__ == "__main__":
    main()