from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from particle import Particle

import phasespace.fromdecay.mass_functions as mf

_kstarz = Particle.from_evtgen_name("K*0")
KSTARZ_MASS = _kstarz.mass
KSTARZ_WIDTH = _kstarz.width


def ref_mass_func(min_mass, max_mass, n_events, key):
    """Reference mass function used to compare the behavior of the actual mass functions.

    Args:
        min_mass: lower limit of mass.
        max_mass: upper limit of mass.
        n_events: number of mass values that should be generated.
        key: JAX PRNG key.

    Returns:
        kstar_mass: Generated mass.

    Notes:
        Code taken from phasespace documentation.
    """
    standard = jax.random.truncated_normal(
        key,
        lower=(min_mass - KSTARZ_MASS) / KSTARZ_WIDTH,
        upper=(max_mass - KSTARZ_MASS) / KSTARZ_WIDTH,
        shape=(n_events,),
        dtype=jnp.float64,
    )
    return KSTARZ_MASS + KSTARZ_WIDTH * standard


@pytest.mark.parametrize(
    "function",
    (mf.gauss_factory, mf.breitwigner_factory, mf.relativistic_breitwigner_factory),
)
@pytest.mark.parametrize("size", (1, 10))
def test_shape(function: Callable, size: int, params: tuple = (1.0, 1.0)):
    key = jax.random.key(1234)
    # the samplers follow the precision of their caller, which is `generate` in normal use
    with jax.enable_x64():
        min_max_mass = jax.random.uniform(key, minval=0, maxval=1000, shape=(2, size), dtype=jnp.float64)
        min_mass, max_mass = jnp.sort(min_max_mass, axis=0)
        assert jnp.all(min_mass <= max_mass)
        ref_sample = ref_mass_func(min_mass, max_mass, len(min_mass), key)
        sample = function(*params)(min_mass, max_mass, len(min_mass), key)
    assert sample.shape == ref_sample.shape


@pytest.mark.parametrize(
    "function",
    (mf.gauss_factory, mf.breitwigner_factory, mf.relativistic_breitwigner_factory),
)
def test_within_limits(function: Callable):
    """The sampled masses have to respect the per-event kinematic limits."""
    n_events = 500
    with jax.enable_x64():
        min_mass = jnp.full((n_events,), 600.0, dtype=jnp.float64)
        max_mass = jnp.full((n_events,), 1200.0, dtype=jnp.float64)
        sample = function(KSTARZ_MASS, KSTARZ_WIDTH)(min_mass, max_mass, n_events, jax.random.key(4))
        assert jnp.all(sample >= min_mass)
        assert jnp.all(sample <= max_mass)


@pytest.mark.parametrize(
    "function",
    (mf.gauss_factory, mf.breitwigner_factory, mf.relativistic_breitwigner_factory),
)
def test_jit(function: Callable):
    """The mass functions run inside jitted generation, so they have to be traceable.

    They are deliberately transparent to the precision mode, which ``generate`` establishes
    around them, so the double precision context is entered here as well.
    """
    n_events = 100
    with jax.enable_x64():
        mass_func = function(KSTARZ_MASS, KSTARZ_WIDTH)
        jitted = jax.jit(lambda lo, hi, key: mass_func(lo, hi, n_events, key))
        sample = jitted(
            jnp.full((n_events,), 600.0, dtype=jnp.float64),
            jnp.full((n_events,), 1200.0, dtype=jnp.float64),
            jax.random.key(0),
        )
    assert sample.shape == (n_events,)
    assert sample.dtype == jnp.float64
