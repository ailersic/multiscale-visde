"""
Variational Inference for Stochastic Differential Equations.

PyTorch Lightning package for learning stochastic differential equations by variational inference.
"""

__version__ = "0.1.0"
__author__ = "Anonymous"

from jaxtyping import install_import_hook

# import protocols and generic classes
from .data import MultiEvenlySpacedTensors, MultiTemporalSampler
from .sde import LatentSDE, LatentSDEConfig, SDE
from .kernel import Kernel
from .sdeprior import LatentDriftConfig, LatentDispersionConfig, LatentDrift, LatentDispersion
from .sdevar import LatentVarConfig, LatentVar, AmortizedLatentVar
from .likelihood import LogLike
from .autoencoder import VarAutoencoderConfig, VarEncoder, VarDecoder

# import specific realizations of protocols
from .kernel import DeepGaussianKernel
from .sdeprior import LatentDriftNoPrior, LatentDispersionNoPrior
from .sdevar import LatentVarGP, AmortizedLatentVarGP, ParamFreeLatentVarGP
from .likelihood import LogLikeGaussian
from .autoencoder import VarEncoderNoPrior, VarDecoderNoPrior
