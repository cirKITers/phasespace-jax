"""Top-level package for JAX PhaseSpace."""

import os
from importlib.metadata import PackageNotFoundError, version

import jax

try:
    __version__ = version("phasespace-jax")
except PackageNotFoundError:
    pass

__author__ = "Melvin Strobl"
__email__ = "melvin.strobl@kit.edu"

# Upstream zfit/phasespace authors, see AUTHORS.md
__credits__ = [
    "Albert Puig Navarro <albert.puig@cern.ch>",
    "Jonas Eschle <jonas.eschle@cern.ch>",
    "Simon Thor",
    "Eduardo Rodrigues <eduardo.rodrigues@cern.ch>",
]

__all__ = ["GenParticle", "nbody_decay", "numpy", "random", "to_vectors"]

import jax.numpy as numpy

from . import random
from .phasespace import GenParticle, nbody_decay, to_vectors


def _set_eager_mode():
    if os.environ.get("PHASESPACE_EAGER", "").lower() not in ("", "0", "false"):
        jax.config.update("jax_disable_jit", True)


_set_eager_mode()
