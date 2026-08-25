"""The generation has to run in double precision without changing global JAX state."""

import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasespace
from phasespace import kinematics as kin

B0_MASS = 5279.58
PION_MASS = 139.57018


def _conservation_residual(parts):
    """Largest relative deviation of the summed daughters from the parent at rest.

    The sum is taken in numpy so that the check itself is always done in double precision,
    independently of the JAX x64 setting.
    """
    total = sum(np.asarray(part, dtype=np.float64) for part in parts.values())
    invariant_mass = np.sqrt(total[:, 3] ** 2 - (total[:, :3] ** 2).sum(axis=1))
    return np.abs(invariant_mass - B0_MASS).max() / B0_MASS


def test_import_does_not_enable_x64_globally():
    """Importing a library must not change the dtype defaults of the calling program."""
    result = subprocess.run(
        [sys.executable, "-c", "import phasespace, jax; print(jax.config.jax_enable_x64)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_generation_does_not_leak_precision_mode():
    with jax.enable_x64(False):
        phasespace.nbody_decay(B0_MASS, [PION_MASS, PION_MASS]).generate(n_events=10, key=1)
        assert not jax.config.jax_enable_x64


@pytest.mark.parametrize("x64", [False, True], ids=["x64_off", "x64_on"])
def test_double_precision_regardless_of_global_flag(x64):
    """The results are float64 and accurate whether or not the caller enabled x64."""
    with jax.enable_x64(x64), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS])
        weights, parts = decay.generate(n_events=2000, key=7)

    assert weights.dtype == np.float64
    assert all(part.dtype == np.float64 for part in parts.values())
    # single precision would both warn on every explicit float64 request and lose ~9 digits
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert _conservation_residual(parts) < 1e-12


def test_kinematics_remain_jit_composable():
    """The low-level helpers must not enter a precision scope of their own.

    Entering one inside a caller's trace mixes dtypes into the jaxpr being built and fails to
    lower, so the helpers stay transparent and follow the precision of their caller.
    """
    with jax.enable_x64(False), warnings.catch_warnings():
        # the helper asks for float64 and JAX warns that it truncates: that is the documented
        # cost of using the low-level helpers outside a precision scope, not a failure
        warnings.simplefilter("ignore", UserWarning)
        vector = jnp.asarray([[1.0, 2.0, 3.0, 10.0]])
        assert jax.jit(kin.mass)(vector).shape == (1, 1)
