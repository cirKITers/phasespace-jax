"""Deterministic checks of Lorentz transformations, including mixed zero boosts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phasespace import kinematics as kin


@pytest.mark.parametrize("compiled", [False, True])
def test_boost_matches_extended_precision_reference(compiled):
    momenta = np.array([[1.0, 2.0, 3.0, 5.0]] * 4)
    boosts = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [1e-10, 0.0, 0.0], [0.0, 0.0, -0.8]])
    p = momenta.astype(np.longdouble)
    b = boosts.astype(np.longdouble)
    gamma = 1 / np.sqrt(1 - np.sum(b * b, axis=1, keepdims=True))
    bp = np.sum(p[:, :3] * b, axis=1, keepdims=True)
    reference = np.concatenate(
        [p[:, :3] + (gamma**2 / (gamma + 1) * bp + gamma * p[:, 3:]) * b, gamma * (p[:, 3:] + bp)],
        axis=1,
    )
    with jax.enable_x64():
        boost = jax.jit(kin.lorentz_boost) if compiled else kin.lorentz_boost
        actual = boost(jnp.asarray(momenta), jnp.asarray(boosts))
        recovered = boost(actual, -jnp.asarray(boosts))
    np.testing.assert_allclose(actual, reference, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(recovered, momenta, rtol=3e-15, atol=3e-15)
    np.testing.assert_array_equal(np.asarray(actual)[0], momenta[0])


def test_beta_is_speed_over_c():
    with jax.enable_x64():
        result = kin.beta(jnp.array([[0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 3.0, 5.0], [3.0, 4.0, 0.0, 5.0]]))
    np.testing.assert_allclose(result[:, 0], [0.0, 0.6, 1.0], rtol=1e-15)
