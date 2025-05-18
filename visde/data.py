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
from torch import Tensor
from torch.utils.data import Dataset, RandomSampler, Sampler
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
        self.M = num_window
        assert torch.allclose(dt, t[:, 1:] - t[:, :-1], atol=1e-4)
        self.dt = dt
        self.mu = mu
        self.t = t
        self.y = y
        self.f = f
        self.n_traj = len(t)
        self.n_data = len(t[0])

    @property
    def total_data(self) -> int:
        return self.n_traj * self.n_data

    def __len__(self) -> int:
        return self.n_traj * (self.n_data - self.M)

    def __getitem__(self, idx: int):
        n_cols = self.n_data - self.M + 1
        traj_id, data_id = idx // n_cols, idx % n_cols
        state_win = self.y[traj_id, data_id : data_id + self.M]
        state = state_win[0]
        forcing = self.f[traj_id, data_id]
        return (self.mu[traj_id],
                self.t[traj_id, data_id],
                state_win,
                state,
                forcing
                )

def break_indices(inds: list[int], M: int) -> list[list[int]]:
    """breaks up a list of ints into a list of lists of length M
    i.e. break_indices([1,2,3,4,5], 2) -> [[1,2], [3,4], [5]]"""
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
    ) -> None:
        self.data_source = data_source
        self.time_window = time_window
        self.generator = generator
        self.n_repeats = n_repeats
        self.sampler = RandomSampler(
            self.indices,
            replacement=False,
            num_samples=len(self.indices) * n_repeats,
            generator=generator,
        )

    @cached_property
    def indices(self) -> list[list[int]]:
        inds = nested_indices(
            self.data_source.n_data - self.data_source.M + 1,
            self.data_source.n_traj,
            self.time_window,
        )
        return [j for i in inds for j in i]

    def __iter__(self):
        yield from (self.indices[i] for i in self.sampler)

    def __len__(self) -> int:
        return len(self.indices) * self.n_repeats
