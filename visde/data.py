"""
Datasets and samplers for amortized reparametrization. All datasets
assume that observations are evenly spaced but this should be easy 
to extend if necessary.
"""

from functools import cached_property

import torch
from jaxtyping import Float, jaxtyped
from beartype import beartype
#from PIL import Image
import h5py as h5
from torch import Tensor
from torch.utils.data import Dataset, RandomSampler, Sampler, SequentialSampler
#from torchvision import transforms
# ruff: noqa: F821, F722

@jaxtyped(typechecker=beartype)
class MultiEvenlySpacedTensors(Dataset):
    mu: Float[Tensor, "n_traj dim_mu"]
    t: Float[Tensor, "n_traj n_tstep"]
    y: Float[Tensor, "n_traj n_tstep ..."]
    dt: Float[Tensor, ""]

    def __init__(
        self,
        mu: Float[Tensor, "n_traj dim_mu"],
        t: Float[Tensor, "n_traj n_tstep"],
        y: Float[Tensor, "n_traj n_tstep ..."],
        f: Float[Tensor, "n_traj n_tstep dim_f"],
        num_window: int,
    ) -> None:
        super().__init__()
        dt = t[0][1] - t[0][0]
        self.n_win = num_window
        assert torch.allclose(dt, t[:, 1:] - t[:, :-1], atol=1e-4)
        self.dt = dt
        self.mu = mu
        self.t = t
        self.y = y
        self.f = f
        self.n_traj = len(t)
        self.n_tsteps = len(t[0])

    @property
    def total_data(self) -> int:
        return self.n_traj * self.n_tsteps

    def __len__(self) -> int:
        return self.n_traj * (self.n_tsteps - self.n_win)

    def __getitem__(self, idx: int):
        n_cols = self.n_tsteps - self.n_win + 1
        traj_id, data_id = idx // n_cols, idx % n_cols
        state_win = self.y[traj_id, data_id : data_id + self.n_win]
        state = state_win[0]
        forcing = self.f[traj_id, data_id]
        return (self.mu[traj_id],
                self.t[traj_id, data_id],
                state_win,
                state,
                forcing
                )

@jaxtyped(typechecker=beartype)
class MultiTrajHDF5(Dataset):
    n_traj: int
    n_tsteps: int

    shape_x: tuple[int, ...]
    dim_mu: int
    dim_f: int
    dt: float

    def __init__(self, h5_path: str, split: str, n_win: int) -> None:
        super().__init__()

        self.h5_path = h5_path
        self.split = split
        self.n_win = n_win

        with h5.File(self.h5_path, 'r') as f:
            self.n_traj = f[f'{split}_x'].shape[0]
            self.n_tsteps = f[f'{split}_x'].shape[1]

            self.shape_x = f[f'{split}_x'].shape[2:]
            self.dim_mu = f[f'{split}_mu'].shape[1]
            self.dim_f = f[f'{split}_f'].shape[1]

            self.dt = f[f'{split}_t'][0, 1] - f[f'{split}_t'][0, 0]

    @property
    def total_data(self) -> int:
        return self.n_traj * self.n_tsteps
    
    def __len__(self) -> int:
        return self.n_traj * (self.n_tsteps - self.n_win + 1)

    def __getitem__(self, idx: int):
        n_cols = self.n_tsteps - self.n_win + 1
        traj_id, data_id = idx // n_cols, idx % n_cols

        # Open HDF5 file lazily inside worker process
        with h5.File(self.h5_path, 'r') as f:
            x_win = f[f'{self.split}_x'][traj_id, data_id : data_id + self.n_win]
            x = x_win[0]
            t = f[f'{self.split}_t'][traj_id, data_id]
            mu = f[f'{self.split}_mu'][traj_id]
            f = f[f'{self.split}_f'][traj_id, data_id]

        return (mu, t, x_win, x, f)

def break_indices(inds: list[int], M: int) -> list[list[int]]:
    """breaks up a list of ints into a list of lists of length M
    i.e. break_indices([1,2,3,4,5,6], 3) -> [[1,2,3],[3,4,5],[5,6]]"""
    lp = 0
    rp = lp + M
    broken_list = []
    while rp < len(inds):
        broken_list.append(inds[lp:rp])
        lp += M - 1 # the -1 gives overlap between segments
        rp += M - 1
    broken_list.append(inds[lp:])
    return broken_list

def nested_indices(n_data, n_traj, M):
    indices = []
    for i in range(n_traj):
        inds = list(range(i * n_data, (i + 1) * n_data))
        indices.append(break_indices(inds, M))
    return indices

@jaxtyped(typechecker=beartype)
class MultiTemporalSampler(Sampler[list[int]]):
    def __init__(
        self,
        data_source: MultiEvenlySpacedTensors,
        time_window: int,
        generator=None,
        n_repeats: int = 1,
        random: bool = True,
    ) -> None:
        self.data_source = data_source
        self.time_window = time_window
        self.generator = generator
        self.n_repeats = n_repeats
        if random:
            self.sampler = RandomSampler(
                self.indices,
                replacement=False,
                num_samples=len(self.indices) * n_repeats,
                generator=generator,
            )
        else:
            self.sampler = SequentialSampler(
                self.indices
            )

    @cached_property
    def indices(self) -> list[list[int]]:
        inds = nested_indices(
            self.data_source.n_tsteps - self.data_source.n_win + 1,
            self.data_source.n_traj,
            self.time_window,
        )
        return [j for i in inds for j in i]

    def __iter__(self):
        yield from (self.indices[i] for i in self.sampler)

    def __len__(self) -> int:
        return len(self.indices) * self.n_repeats
