"""Top-level package for JAX PhaseSpace."""

import os
from importlib.metadata import PackageNotFoundError, version

import jax

try:
    __version__ = version("phasespace-jax")
except PackageNotFoundError:
    pass

__author__ = """Albert Puig Navarro"""
__email__ = "apuignav@gmail.com"
__maintainer__ = "Melvin Strobl <stroblme@posteo.de>"

__credits__ = [
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
