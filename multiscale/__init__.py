"""
Learning multiscale stochastic differential equations by variational inference.
"""

__version__ = "0.1.0"
__author__ = "Andrew Francesco Ilersich, Prasanth B. Nair"

from jaxtyping import install_import_hook

# import protocols and generic classes
from .autoencoder import MultiscaleVarEncoderNoPrior, MultiscaleVarDecoderNoPrior
from .sdeprior import MultiscaleLatentDriftNoPrior, MultiscaleLatentDispersionNoPrior