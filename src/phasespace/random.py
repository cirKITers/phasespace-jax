"""Random number generation.

JAX random number generation is purely functional: every draw is an explicit function of a
PRNG key. This module only normalizes what users may pass as a key.
"""

from __future__ import annotations

import secrets

import jax
import numpy as np

KeyLike = int | jax.Array | None


def ensure_key(key: KeyLike = None) -> jax.Array:
    """Normalize a user-supplied key into a JAX PRNG key.

    Args:
        key: This can be
          - `None` to create a new, non-reproducible key from OS entropy,
          - an integer seed to create a key deterministically,
          - a JAX PRNG key, which is returned unchanged.

    Returns:
        A JAX PRNG key.

    Notes:
        Never call this with `None` inside a jitted function: the key would be created once at
        trace time and every call would then reuse the very same random numbers.
    """
    if key is None:
        return jax.random.key(secrets.randbits(63))
    if isinstance(key, (int, np.integer)):
        return jax.random.key(int(key))
    if isinstance(key, (jax.Array, np.ndarray)):
        return key  # a PRNG key; jax.random validates it on use
    raise TypeError(f"Expected an int seed, a JAX PRNG key or None, got {type(key).__name__}.")
