import jax
import numpy as np
import pytest

import phasespace as phsp


@pytest.mark.parametrize("key", [lambda: 15, lambda: jax.random.key(15)], ids=["int", "key"])
def test_ensure_key(key):
    key1 = phsp.random.ensure_key(key())
    key2 = phsp.random.ensure_key(key())
    rnd1 = jax.random.uniform(key1, shape=(100,))
    rnd2 = jax.random.uniform(key2, shape=(100,))

    # the same seed or key always gives the same numbers, JAX generation is purely functional
    np.testing.assert_array_equal(rnd1, rnd2)

    rnd_unseeded = jax.random.uniform(phsp.random.ensure_key(), shape=(100,))
    assert not np.array_equal(rnd1, rnd_unseeded)


def test_ensure_key_passthrough():
    key = jax.random.key(15)
    assert phsp.random.ensure_key(key) is key


def test_ensure_key_unseeded_differs():
    rnd1 = jax.random.uniform(phsp.random.ensure_key(), shape=(100,))
    rnd2 = jax.random.uniform(phsp.random.ensure_key(), shape=(100,))
    assert not np.array_equal(rnd1, rnd2)


def test_ensure_key_invalid():
    with pytest.raises(TypeError):
        phsp.random.ensure_key(1.5)
