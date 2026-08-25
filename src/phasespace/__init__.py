"""Top-level package for JAX PhaseSpace."""

import os
from importlib.metadata import PackageNotFoundError, version

import jax

# Double precision is required: `pdk` suffers catastrophic cancellation close to threshold.
# NOTE: this is a process-wide JAX setting.
jax.config.update("jax_enable_x64", True)

try:
    __version__ = version("phasespace-jax")
except PackageNotFoundError:
    pass

__author__ = """Albert Puig Navarro"""
__email__ = "apuignav@gmail.com"
__maintainer__ = "zfit"

__credits__ = ["Jonas Eschle <Jonas.Eschle@cern.ch>"]

__all__ = ["GenParticle", "nbody_decay", "numpy", "random", "to_vectors"]

import jax.numpy as numpy  # noqa: E402

from . import random  # noqa: E402
from .phasespace import GenParticle, nbody_decay, to_vectors  # noqa: E402


def _set_eager_mode():
    if os.environ.get("PHASESPACE_EAGER", "").lower() not in ("", "0", "false"):
        jax.config.update("jax_disable_jit", True)


_set_eager_mode()
