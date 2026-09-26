from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from particle import Particle
from scipy.integrate import quad
from scipy.optimize import brentq

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


@pytest.mark.parametrize("limits", [(600.0, 1200.0), (50.0, 200.0), (600.0, 600.0 + 1e-6)])
def test_relbw_tail_respects_limits_and_is_reproducible(limits):
    with jax.enable_x64():
        sampler = mf.relativistic_breitwigner_factory(1.0, 1.0)
        low, high = (jnp.full((1000,), bound) for bound in limits)
        call = jax.jit(lambda key: sampler(low, high, 1000, key))
        first = np.asarray(call(jax.random.key(7)))
        second = np.asarray(call(jax.random.key(7)))
    assert np.all(np.isfinite(first))
    assert np.all(first >= limits[0])
    assert np.all(first <= limits[1])
    assert np.ptp(first) > 0
    np.testing.assert_array_equal(first, second)


def test_relbw_tail_does_not_change_other_events():
    with jax.enable_x64():
        sampler = mf.relativistic_breitwigner_factory(1.0, 1.0)
        low = jnp.zeros(100)
        high = jnp.full((100,), 2.0)
        key = jax.random.key(7)
        original = np.asarray(sampler(low, high, 100, key))
        mixed = np.asarray(sampler(low.at[-1].set(600.0), high.at[-1].set(1200.0), 100, key))
    np.testing.assert_array_equal(original[:-1], mixed[:-1])
    assert 600.0 <= mixed[-1] <= 1200.0


def test_relbw_uses_independent_random_keys():
    with jax.enable_x64(), jax.debug_key_reuse(True):
        sampler = mf.relativistic_breitwigner_factory(1.0, 1.0)
        sample = jax.jit(lambda key: sampler(600.0, 1200.0, 5, key))(jax.random.key(7))
    assert np.all(np.isfinite(sample))


@pytest.mark.parametrize("limits", [(0.5, 1.5), (600.0, 1200.0)])
@pytest.mark.parametrize("shape", [(), (1,), (5,)])
def test_relbw_broadcast_limits_and_derivatives(limits, shape):
    """The fallback preserves broadcasting and derivatives of fixed-key samples."""
    with jax.enable_x64():
        sampler = mf.relativistic_breitwigner_factory(1.0, 1.0)
        bounds = jnp.asarray(limits)

        def draw(bounds):
            return sampler(jnp.full(shape, bounds[0]), jnp.full(shape, bounds[1]), 5, jax.random.key(7))

        sample = jax.jit(draw)(bounds)
        derivative = jax.jit(jax.jacrev(draw))(bounds)
        step = 1e-5
        numerical = jnp.stack(
            [
                (draw(bounds + step * direction) - draw(bounds - step * direction)) / (2 * step)
                for direction in jnp.eye(2)
            ],
            axis=1,
        )
    sample = np.asarray(sample)
    assert sample.shape == (5,)
    assert np.all((sample >= limits[0]) & (sample <= limits[1]))
    np.testing.assert_allclose(derivative, numerical, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "mass,width,low,high",
    [(1.0, 1.0, 600.0, 1200.0), (1.0, 1.0, 50.0, 200.0), (KSTARZ_MASS, KSTARZ_WIDTH, 0.0, 30000.0)],
)
def test_relbw_tail_distribution_matches_density(mass, width, low, high):
    """Check quantiles against independent quadrature of the upstream density."""
    scale = (np.clip(mass, low, high) ** 2 - mass**2) ** 2 + (mass * width) ** 2

    def density(x):
        return scale / ((x**2 - mass**2) ** 2 + (mass * width) ** 2)

    norm = quad(density, low, high, epsabs=1e-11, epsrel=1e-11)[0]
    n = 20000
    with jax.enable_x64():
        sampler = mf.relativistic_breitwigner_factory(mass, width)
        sample = np.asarray(
            jax.jit(lambda key: sampler(jnp.full((n,), low), jnp.full((n,), high), n, key))(jax.random.key(13))
        )
    for probability in (0.1, 0.5, 0.9):
        quantile = brentq(
            lambda x, probability=probability: (
                quad(density, low, x, epsabs=1e-11, epsrel=1e-11)[0] / norm - probability
            ),
            low,
            high,
        )
        observed = np.mean(sample <= quantile)
        assert abs(observed - probability) < 6 * np.sqrt(probability * (1 - probability) / n)
