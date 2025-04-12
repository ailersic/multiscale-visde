from jaxtyping import Float
import torch
from torch import Tensor
import pickle as pkl
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np

# ruff: noqa: F821, F722

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

def dyn(mu: Float[Tensor, "n_traj dim_mu"],
        t: Float[Tensor, "n_traj"],
        x: Float[Tensor, "n_traj dim_x"],
        f: Float[Tensor, "n_traj dim_f"]
) -> Float[Tensor, "n_traj dim_x"]:
    nu = mu[:, 0:1]

    dim_x = x.shape[-1]
    delta_x = 1/dim_x # not dim_x - 1 because boundary conditions are periodic

    ip1 = (np.arange(dim_x) + 1) % dim_x
    im1 = (np.arange(dim_x) - 1) % dim_x

    xp1 = x[:, ip1]
    xm1 = x[:, im1]
    dx = (xp1 - xm1)/(2*delta_x)
    d2x = (xp1 - 2*x + xm1)/(delta_x**2)

    dxdt = -x*dx + nu*d2x

    return dxdt

def forcing(t: Float[Tensor, "n_tstep"], dim_f: int
) -> Float[Tensor, "n_tstep dim_f"]:
    t_ = t.unsqueeze(1)
    f = torch.zeros(t_.shape[0], dim_f)
    return f

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
        amp = 0.25 + 0.05 * torch.randn(1)
        var = 0.01 + 0.0025 * torch.randn(1)
        #offset = 0.25 + 0.1 * torch.randn(1)
        x[i, 0, :] = amp*torch.exp(-(x_domain - 0.5)**2/var)
        f[i] = forcing(t[i, :], dim_f)
        plt.plot(x_domain.numpy(), x[i, 0, :].numpy())
    plt.show()

    for i in range(1, n_tstep):
        dt = (t[:, i] - t[:, i - 1]).unsqueeze(1)
        xi1 = x[:, i - 1] + dyn(mu, t[:, i - 1], x[:, i - 1], f[:, i - 1]) * dt
        xi2 = x[:, i - 1] + dyn(mu, t[:, i - 1], xi1, f[:, i - 1]) * dt
        x[:, i] = 0.5 * (xi1 + xi2)
    
    return t, x, f

def main():
    n_traj = 10
    n_tstep = 1001

    dim_x = 1000

    T = 1.0
    mu = torch.cat([torch.full([n_traj, 1], 0.0005)], dim=1)

    t, x, f = create_dataset(mu, T, n_tstep, dim_x)
    x += 0.001 * torch.randn_like(x)

    for i in range(n_traj):
        plt.figure()
        plt.plot(np.linspace(0, 1, dim_x), x[i, 0, :dim_x].numpy(), label="t=0")
        plt.plot(np.linspace(0, 1, dim_x), x[i, n_tstep//4, :dim_x].numpy(), label="t=T/4")
        plt.plot(np.linspace(0, 1, dim_x), x[i, n_tstep//2, :dim_x].numpy(), label="t=T/2")
        plt.plot(np.linspace(0, 1, dim_x), x[i, n_tstep*3//4, :dim_x].numpy(), label="t=3T/4")
        plt.plot(np.linspace(0, 1, dim_x), x[i, -1, :dim_x].numpy(), label="t=T")
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(CURR_DIR, f"x{i}.png"))

    #i_train = range(0, 1001)
    #i_val = range(1001, 1251)
    #i_test = range(1251, 1501)

    n_traj_train = 8
    n_traj_val = 1
    n_traj_test = 1

    train_mu, train_t, train_x, train_f = mu[:n_traj_train], t[:n_traj_train], x[:n_traj_train], f[:n_traj_train]
    val_mu, val_t, val_x, val_f = mu[n_traj_train:n_traj_train + n_traj_val], t[n_traj_train:n_traj_train + n_traj_val], x[n_traj_train:n_traj_train + n_traj_val], f[n_traj_train:n_traj_train + n_traj_val]
    test_mu, test_t, test_x, test_f = mu[n_traj_train + n_traj_val:], t[n_traj_train + n_traj_val:], x[n_traj_train + n_traj_val:], f[n_traj_train + n_traj_val:]

    assert train_mu.shape == (n_traj_train, 1)
    assert train_t.shape == (n_traj_train, n_tstep)
    assert train_x.shape == (n_traj_train, n_tstep, dim_x)
    assert train_f.shape == (n_traj_train, n_tstep, 1)

    assert val_mu.shape == (n_traj_val, 1)
    assert val_t.shape == (n_traj_val, n_tstep)
    assert val_x.shape == (n_traj_val, n_tstep, dim_x)
    assert val_f.shape == (n_traj_val, n_tstep, 1)

    assert test_mu.shape == (n_traj_test, 1)
    assert test_t.shape == (n_traj_test, n_tstep)
    assert test_x.shape == (n_traj_test, n_tstep, dim_x)
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

    with open(os.path.join(CURR_DIR, "data.pkl"), "wb") as f:
        pkl.dump(data, f)

if __name__ == "__main__":
    main()