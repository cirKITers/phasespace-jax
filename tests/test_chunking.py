"""Chunked generation has to agree with an unchunked one in everything but the sample it draws."""

import numpy as np
import pytest

import phasespace

B0_MASS = 5279.58
PION_MASS = 139.57018


def _decay(n_pions=3):
    return phasespace.nbody_decay(B0_MASS, [PION_MASS] * n_pions)


def _total(parts):
    """Summed daughter 4-momentum.

    The sum is taken in numpy so that the check itself is always done in double precision,
    independently of the JAX x64 setting.
    """
    return sum(np.asarray(part, dtype=np.float64) for part in parts.values())


def _boosted_along_z(pz):
    """A ``B0_MASS`` 4-momentum per entry of ``pz``, kept as numpy so no precision is lost."""
    pz = np.asarray(pz, dtype=np.float64)
    zero = np.zeros_like(pz)
    return np.stack([zero, zero, pz, np.sqrt(pz**2 + B0_MASS**2)], axis=-1)


@pytest.mark.parametrize("chunk_size", [1000, 2000], ids=["exactly_n_events", "above_n_events"])
def test_chunk_size_of_at_least_n_events_is_a_no_op(chunk_size):
    """A chunk that covers everything must take the unchunked path, hence give identical numbers."""
    weights_ref, parts_ref = _decay().generate(1000, key=3)
    weights, parts = _decay().generate(1000, key=3, chunk_size=chunk_size)

    np.testing.assert_array_equal(weights, weights_ref)
    for name, part in parts.items():
        np.testing.assert_array_equal(part, parts_ref[name])


def test_remainder_chunk_completes_the_sample():
    """1000 events in chunks of 300 is 300 + 300 + 300 + 100, not 900."""
    weights, parts = _decay().generate(1000, key=11, chunk_size=300)

    assert weights.shape == (1000,)
    assert all(part.shape == (1000, 4) for part in parts.values())
    assert np.all(np.isfinite(np.asarray(weights)))
    assert all(np.all(np.isfinite(np.asarray(part))) for part in parts.values())


def test_chunking_keeps_double_precision_and_conserves_energy_momentum():
    weights, parts = _decay().generate(2000, key=13, chunk_size=512)

    assert weights.dtype == np.float64
    assert all(part.dtype == np.float64 for part in parts.values())

    total = _total(parts)
    invariant_mass = np.sqrt(total[:, 3] ** 2 - (total[:, :3] ** 2).sum(axis=1))
    assert np.abs(invariant_mass - B0_MASS).max() / B0_MASS < 1e-12


def test_chunking_is_reproducible():
    first = _decay().generate(700, key=17, chunk_size=256)
    second = _decay().generate(700, key=17, chunk_size=256)

    np.testing.assert_array_equal(first[0], second[0])
    for name, part in first[1].items():
        np.testing.assert_array_equal(part, second[1][name])


def test_a_different_chunk_size_draws_a_different_sample():
    """Documented behaviour: every chunk consumes its own split of the key."""
    weights_one, _ = _decay().generate(700, key=17, chunk_size=256)
    weights_other, _ = _decay().generate(700, key=17, chunk_size=128)

    assert not np.allclose(np.asarray(weights_one), np.asarray(weights_other))


def test_unnormalized_chunking_returns_the_max_weights():
    weights, weights_max, _ = _decay().generate(500, key=19, normalize_weights=False, chunk_size=128)

    assert weights.shape == (500,)
    assert weights_max.shape == (500,)
    # normalizing the stitched result is what the normalized call does, so it must agree exactly
    normalized, _ = _decay().generate(500, key=19, chunk_size=128)
    np.testing.assert_array_equal(normalized, weights / weights_max)


def test_per_event_boost_to_is_sliced_per_chunk():
    """The daughters of event i have to add up to the momentum event i was boosted to.

    A chunk handed the wrong slice of ``boost_to`` would leave the later events boosted to the
    momentum of the first chunk, which this catches.
    """
    n_events = 500
    # every event is boosted: a batch mixing zero and non-zero boosts hits an unrelated NaN in
    # `kinematics.lorentz_boost`, which decides on the zero-boost shortcut for the whole batch at once
    boost_to = _boosted_along_z(np.linspace(1000.0, 20000.0, n_events))

    _, parts = _decay().generate(n_events, boost_to=boost_to, key=5, chunk_size=128)

    np.testing.assert_allclose(_total(parts), boost_to, rtol=1e-12, atol=1e-6)


def test_broadcast_boost_to_reaches_every_chunk():
    """A single 4-momentum applies to all events, so it is handed to every chunk unsliced."""
    boost_to = _boosted_along_z([12000.0])

    _, parts = _decay().generate(500, boost_to=boost_to, key=23, chunk_size=128)

    np.testing.assert_allclose(_total(parts), np.broadcast_to(boost_to, (500, 4)), rtol=1e-12, atol=1e-6)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_is_rejected(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        _decay().generate(100, key=1, chunk_size=chunk_size)


def test_chunking_is_available_through_genmultidecay():
    """``GenMultiDecay`` forwards kwargs to ``GenParticle.generate``, so ``chunk_size`` flows through."""
    genmultidecay = pytest.importorskip("phasespace.fromdecay.genmultidecay")
    example_decay_chains = pytest.importorskip("tests.fromdecay.example_decay_chains")

    decay = genmultidecay.GenMultiDecay.from_dict(example_decay_chains.dplus_single)
    weights, _ = decay.generate(1000, key=29, chunk_size=128)

    assert sum(len(w) for w in weights) == 1000
    assert all(np.all(np.isfinite(np.asarray(w))) for w in weights)


if __name__ == "__main__":
    pytest.main([__file__])
