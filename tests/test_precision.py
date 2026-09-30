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


@pytest.mark.parametrize(
    "mass,masses",
    [(5.0, [0.0, 0.0, 0.0]), (5.0, [0.0, 0.0, 0.0, 0.0]), (5.0, [0.0, 1.0, 2.0]), (3.0 + 1e-10, [1.0, 1.0, 1.0])],
)
def test_four_momentum_and_mass_shells(mass, masses):
    """The massless seed includes an intermediate system with gamma above 2e5."""
    weights, particles = phasespace.nbody_decay(mass, masses).generate(2000, key=7)
    arrays = [np.asarray(p) for p in particles.values()]
    assert np.all(np.isfinite(weights))
    assert np.all((np.asarray(weights) >= 0) & (np.asarray(weights) <= 1 + 1e-14))
    np.testing.assert_allclose(
        sum(arrays), np.broadcast_to([0.0, 0.0, 0.0, mass], (2000, 4)), rtol=1e-13, atol=1e-13 * mass
    )
    for p, daughter_mass in zip(arrays, masses):
        np.testing.assert_allclose(
            p[:, 3] ** 2 - np.sum(p[:, :3] ** 2, axis=1), daughter_mass**2, rtol=1e-13, atol=1e-13 * mass**2
        )
        assert np.all(p[:, 3] >= 0)
        assert p.dtype == np.float64


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("chunk", [None, 8])
def test_threshold_decay_is_rejected(normalize, chunk):
    decay = phasespace.nbody_decay(3.0, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="no positive available phase space"):
        decay.generate(20, key=3, normalize_weights=normalize, chunk_size=chunk)


def test_conservation_at_each_decay_vertex():
    from .helpers.decays import bp_to_k1_kstar_pi_gamma

    weights, parts = bp_to_k1_kstar_pi_gamma().generate(2000, key=7)
    parts = {name: np.asarray(p) for name, p in parts.items()}
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(parts["K+"] + parts["pi-"], parts["K*0"], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(parts["K*0"] + parts["pi+"], parts["K1+"], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        parts["K1+"] + parts["gamma"], np.broadcast_to([0.0, 0.0, 0.0, B0_MASS], (2000, 4)), rtol=1e-12, atol=1e-9
    )


def test_massless_three_body_energy_and_angles():
    """Constant matrix element: dPhi3 is uniform on the massless Dalitz triangle.

    Thus x = 2E/M has density 2x on [0, 1]; directions are isotropic.
    See PDG 2025, Kinematics, Sec. 49.4.3:
    https://pdg.lbl.gov/2025/reviews/rpp2025-rev-kinematics.pdf
    """
    weights, parts = phasespace.nbody_decay(5.0, [0.0, 0.0, 0.0]).generate(50000, key=11)
    weights = np.asarray(weights)
    for p in parts.values():
        p = np.asarray(p)
        x = 2 * p[:, 3] / 5
        cos_theta = p[:, 2] / p[:, 3]
        for observable, expected in ((x, 2 / 3), (x * x, 1 / 2), (cos_theta, 0.0), (cos_theta**2, 1 / 3)):
            mean = np.sum(weights * observable) / np.sum(weights)
            error = np.sqrt(np.sum(weights**2 * (observable - expected) ** 2)) / np.sum(weights)
            assert abs(mean - expected) < 6 * error


def test_decay_stage_supports_jax_transformations():
    """Sharing rotation coefficients must preserve batching and mass derivatives."""
    with jax.enable_x64():

        def momenta(scale, key):
            particles, _ = phasespace.GenParticle._generate_part2(
                [jnp.full((4, 1), mass) * scale for mass in (1.0, 3.0, 6.0)],
                jnp.ones((4, 3)) * scale,
                4,
                3,
                key,
            )
            return jnp.stack(particles)

        keys = jax.random.split(jax.random.key(7), 3)
        batched = jax.jit(jax.vmap(momenta, in_axes=(None, 0)))(1.0, keys)
        np.testing.assert_allclose(
            np.asarray(batched).sum(axis=1), np.broadcast_to([0.0, 0.0, 0.0, 6.0], (3, 4, 4)), atol=1e-13
        )
        derivative = jax.jit(jax.grad(lambda scale: momenta(scale, keys[0])[..., 3].sum()))(1.0)
        np.testing.assert_allclose(derivative, 24.0, rtol=1e-13)
