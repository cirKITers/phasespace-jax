"""``sort_rows`` replaces ``jnp.sort`` on the short axis, so it has to sort exactly like it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phasespace.phasespace import sort_rows

N_ROWS = 2000


def _inputs(n_columns, seed):
    """A few input families, chosen so ties and duplicates are the common case, not the exception."""
    rng = np.random.default_rng(seed)
    shape = (N_ROWS, n_columns)
    return {
        "uniform": jax.random.uniform(jax.random.key(seed), shape, dtype=jnp.float64),
        "ties": jnp.asarray(rng.integers(0, 3, shape), dtype=jnp.float64),
        "all_equal": jnp.full(shape, 0.5, dtype=jnp.float64),
        "descending": jnp.broadcast_to(jnp.arange(n_columns, 0, -1, dtype=jnp.float64), shape),
        # the ends of the interval `jax.random.uniform` draws from; subnormals are deliberately
        # left out, as the CPU backend flushes those to zero in `minimum` but not in `sort`
        "edges": jnp.asarray(rng.choice([0.0, 1.0, np.nextafter(1.0, 0.0), 2.0**-53], shape), dtype=jnp.float64),
    }


@pytest.mark.parametrize("n_columns", range(9))
def test_matches_jnp_sort_exactly(n_columns):
    """Bit-exact, not merely close: the generation must not shift because of this helper."""
    with jax.enable_x64():
        for name, values in _inputs(n_columns, seed=n_columns).items():
            np.testing.assert_array_equal(
                np.asarray(sort_rows(values, n_columns)),
                np.asarray(jnp.sort(values, axis=1)),
                err_msg=f"{name} input with {n_columns} column(s)",
            )


@pytest.mark.parametrize("n_columns", [0, 1, 2, 5])
def test_matches_jnp_sort_under_jit(n_columns):
    """It is only ever called from inside a jitted function."""
    with jax.enable_x64():
        values = jax.random.uniform(jax.random.key(7), (N_ROWS, n_columns), dtype=jnp.float64)
        sorted_rows = jax.jit(sort_rows, static_argnums=1)(values, n_columns)

        np.testing.assert_array_equal(np.asarray(sorted_rows), np.asarray(jnp.sort(values, axis=1)))


@pytest.mark.parametrize("n_columns", [0, 1, 4])
def test_shape_and_dtype_are_preserved(n_columns):
    with jax.enable_x64():
        values = jax.random.uniform(jax.random.key(3), (N_ROWS, n_columns), dtype=jnp.float64)
        sorted_rows = sort_rows(values, n_columns)

        assert sorted_rows.shape == values.shape
        assert sorted_rows.dtype == values.dtype


if __name__ == "__main__":
    pytest.main([__file__])
