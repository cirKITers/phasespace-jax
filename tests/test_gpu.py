"""The generation has to run on a GPU without losing double precision or changing the physics.

These tests skip themselves when no GPU backend is available, which is the case on CI: GitHub
provides no GPU runners, so this file is meant to be run locally against a CUDA-enabled jaxlib
(``uv pip install "jax[cuda12]"`` for Pascal/Volta, ``jax[cuda13]`` for SM 7.5 and newer).
"""

import warnings

import jax
import numpy as np
import pytest

import phasespace

from .test_precision import B0_MASS, PION_MASS, _conservation_residual


def _devices(platform):
    try:
        return jax.devices(platform)
    except RuntimeError:  # backend not built into this jaxlib, or excluded by JAX_PLATFORMS
        return []


_GPUS = _devices("gpu")
_CPUS = _devices("cpu")

requires_gpu = pytest.mark.skipif(not _GPUS, reason="no GPU backend available")
# JAX_PLATFORMS=cuda leaves cuda as the only backend, so a cross-backend comparison cannot run
requires_both = pytest.mark.skipif(not (_GPUS and _CPUS), reason="needs both a CPU and a GPU backend")

# CPU and GPU differ by a few ULP per operation, mostly through the transcendentals and the
# cancellation in `pdk`.
MOMENTUM_RTOL, MOMENTUM_ATOL = 1e-12, 1e-9
WEIGHT_RTOL = 1e-10


def _decay(n_pions=3):
    return phasespace.nbody_decay(B0_MASS, [PION_MASS] * n_pions)


def _generate_on(device, n_events=20000, **kwargs):
    with jax.default_device(device):
        return _decay().generate(n_events, key=42, **kwargs)


@requires_gpu
def test_generation_runs_on_the_gpu():
    _, parts = _generate_on(_GPUS[0], n_events=1000)

    assert all(part.devices() == {_GPUS[0]} for part in parts.values())


@requires_gpu
def test_double_precision_survives_on_the_gpu():
    """A GPU backend that silently fell back to float32 would be caught here, not in the physics."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        weights, parts = _generate_on(_GPUS[0], n_events=2000)

    assert weights.dtype == np.float64
    assert all(part.dtype == np.float64 for part in parts.values())
    # single precision would both warn on every explicit float64 request and lose ~9 digits
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert _conservation_residual(parts) < 1e-12


@requires_both
def test_cpu_and_gpu_agree_for_the_same_key():
    """The strongest check available: same seed, same events, up to floating-point noise.

    It covers both halves of portability at once, since the threefry PRNG is defined to be
    backend-independent: a disagreement in the random draws would show up as an O(1) difference
    rather than the ULP-level one the tolerances allow.
    """
    weights_cpu, parts_cpu = _generate_on(_CPUS[0])
    weights_gpu, parts_gpu = _generate_on(_GPUS[0])

    np.testing.assert_allclose(np.asarray(weights_gpu), np.asarray(weights_cpu), rtol=WEIGHT_RTOL)
    for name, part in parts_cpu.items():
        np.testing.assert_allclose(
            np.asarray(parts_gpu[name], dtype=np.float64),
            np.asarray(part, dtype=np.float64),
            rtol=MOMENTUM_RTOL,
            atol=MOMENTUM_ATOL,
        )


@requires_gpu
def test_chunked_generation_on_the_gpu():
    """Chunking exists for GPU memory, so it has to hold up on the device it was added for."""
    weights, parts = _generate_on(_GPUS[0], n_events=5000, chunk_size=1024)

    assert weights.shape == (5000,)
    assert all(part.devices() == {_GPUS[0]} for part in parts.values())
    assert all(part.dtype == np.float64 for part in parts.values())
    assert _conservation_residual(parts) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__])
