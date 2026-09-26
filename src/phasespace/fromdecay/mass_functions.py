"""Mass distribution functions for resonant particles.

This module provides factory functions that create mass distribution functions for resonant
particles. Each factory returns a callable with the signature
``(min_mass, max_mass, n_events, key)`` that samples masses truncated to ``[min_mass, max_mass]``
and is usable inside jitted code.

Sampling is done by inverse transform sampling (see e.g. L. Devroye, *Non-Uniform Random Variate
Generation*, Springer 1986, Ch. II), analytically where a closed-form quantile function exists and
on a precomputed grid for the relativistic Breit-Wigner, which has none. A bounded rejection
sampler covers nonnegative kinematic intervals that the table cannot resolve.
"""

import jax
import jax.numpy as jnp
import numpy as np

#: Number of grid points used to tabulate the relativistic Breit-Wigner CDF.
_RELBW_GRID_POINTS = 20000


def gauss_factory(mass, width):
    """Create a Gaussian mass distribution function.

    Args:
        mass: Mean mass of the particle.
        width: Width (sigma) of the Gaussian distribution.

    Returns:
        Callable that generates masses from a Gaussian distribution truncated to
        ``[min_mass, max_mass]``, with signature ``(min_mass, max_mass, n_events, key)`` and
        returning an array of shape ``(n_events,)``.
    """
    particle_mass = float(mass)
    particle_width = float(width)

    def gauss(min_mass, max_mass, n_events, key):
        # jax.random.truncated_normal samples the *standard* normal restricted to the bounds,
        # so the bounds are standardized and the samples scaled back afterwards.
        standard = jax.random.truncated_normal(
            key,
            lower=(min_mass - particle_mass) / particle_width,
            upper=(max_mass - particle_mass) / particle_width,
            shape=(n_events,),
            dtype=jnp.float64,
        )
        return particle_mass + particle_width * standard

    return gauss


def breitwigner_factory(mass, width):
    """Create a Breit-Wigner (Cauchy) mass distribution function.

    Args:
        mass: Central mass (m) of the particle.
        width: Width (gamma) of the Breit-Wigner distribution.

    Returns:
        Callable that generates masses from a Breit-Wigner distribution truncated to
        ``[min_mass, max_mass]``, with signature ``(min_mass, max_mass, n_events, key)`` and
        returning an array of shape ``(n_events,)``.

    Notes:
        The Cauchy CDF is :math:`F(x) = 1/2 + \\arctan((x - m) / \\gamma) / \\pi`, which is inverted
        analytically to sample within the limits.
    """
    particle_mass = float(mass)
    particle_width = float(width)

    def cdf(x):
        return 0.5 + jnp.arctan((x - particle_mass) / particle_width) / jnp.pi

    def bw(min_mass, max_mass, n_events, key):
        uniform = jax.random.uniform(key, (n_events,), dtype=jnp.float64)
        cdf_low = cdf(min_mass)
        quantile = cdf_low + uniform * (cdf(max_mass) - cdf_low)
        return particle_mass + particle_width * jnp.tan(jnp.pi * (quantile - 0.5))

    return bw


def relativistic_breitwigner_factory(mass, width):
    """Create a relativistic Breit-Wigner mass distribution function.

    Args:
        mass: Central mass (m) of the particle.
        width: Width (gamma) of the relativistic Breit-Wigner distribution.

    Returns:
        Callable that generates masses from a relativistic Breit-Wigner distribution truncated to
        ``[min_mass, max_mass]``, with signature ``(min_mass, max_mass, n_events, key)`` and
        returning an array of shape ``(n_events,)``.

    Notes:
        The density is the constant-width relativistic Breit-Wigner
        :math:`f(m) \\propto 1 / ((m^2 - m_0^2)^2 + m_0^2 \\Gamma^2)` (PDG, Review of Particle
        Physics, resonance section), matching ``zfit_physics.pdf.RelativisticBreitWigner``.

        It has no closed-form quantile function, so the CDF is tabulated once here and inverted by
        interpolation. The grid is placed at the quantiles of the Cauchy distribution that the
        density takes in :math:`s = m^2`, which makes it dense across the peak while still reaching
        far into the tails. Intervals outside the table, or too narrow to resolve in its CDF,
        use rejection sampling of the same density instead of clipping the requested limits.
    """
    particle_mass = float(mass)
    particle_width = float(width)

    theta = np.linspace(
        np.arctan(-particle_mass / particle_width),  # s = 0
        np.pi / 2,
        _RELBW_GRID_POINTS + 1,
    )[:-1]
    s = particle_mass**2 + particle_mass * particle_width * np.tan(theta)
    grid_mass = np.sqrt(np.clip(s, 0.0, None))
    density = 1.0 / ((grid_mass**2 - particle_mass**2) ** 2 + particle_mass**2 * particle_width**2)
    grid_cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid_mass))])
    grid_cdf /= grid_cdf[-1]
    # kept as numpy: converting here would pin the dtype outside the caller's float64 scope

    # Factor the denominator as ((x - a)**2 + b**2) * ((x + a)**2 + b**2),
    # where a**2 - b**2 = m**2 and 2*a*b = m*width. A Cauchy(a, b) proposal
    # truncated to [lo, hi] then has acceptance probability
    # ((lo + a)**2 + b**2) / ((x + a)**2 + b**2) for nonnegative mass limits.
    a = np.sqrt((np.hypot(particle_mass**2, particle_mass * particle_width) + particle_mass**2) / 2)
    b = particle_mass * particle_width / (2 * a)

    def relbw(min_mass, max_mass, n_events, key):
        # An independent fallback stream retains the existing in-table random draws.
        tail_key = jax.random.fold_in(key, 1)
        uniform = jax.random.uniform(key, (n_events,), dtype=jnp.float64)
        cdf_low = jnp.interp(min_mass, grid_mass, grid_cdf)
        cdf_high = jnp.interp(max_mass, grid_mass, grid_cdf)
        quantile = cdf_low + uniform * (cdf_high - cdf_low)
        sample = jnp.interp(quantile, grid_cdf, grid_mass)
        outside = (min_mass < grid_mass[0]) | (max_mass > grid_mass[-1]) | (cdf_high <= cdf_low)
        outside = jnp.broadcast_to(outside, sample.shape)

        def sample_tail():
            # Complementary Cauchy angles avoid subtracting CDF values close to one.
            angle_low = jnp.arctan2(b, min_mass - a)
            angle_high = jnp.arctan2(b, max_mass - a)
            envelope = (min_mass + a) ** 2 + b**2

            def draw(state):
                key, selected_uniform, accepted = state
                key, proposal_key, accept_key = jax.random.split(key, 3)
                uniform = jax.random.uniform(proposal_key, (n_events,), dtype=jnp.float64)
                angle = angle_low + uniform * (angle_high - angle_low)
                proposal = jnp.clip(a + b / jnp.tan(angle), min_mass, max_mass)
                accept = jax.random.uniform(accept_key, (n_events,), dtype=jnp.float64)
                accept = accept * ((proposal + a) ** 2 + b**2) <= envelope
                selected_uniform = jnp.where(~accepted & accept, uniform, selected_uniform)
                return key, selected_uniform, accepted | accept

            selected_uniform = jax.lax.while_loop(
                lambda state: jnp.any(~state[2]), draw, (tail_key, uniform, ~outside)
            )[1]
            # The discrete acceptance decisions have no derivative. Evaluate the selected
            # proposal outside the loop to retain derivatives with respect to the mass limits.
            angle = angle_low + jax.lax.stop_gradient(selected_uniform) * (angle_high - angle_low)
            proposal = jnp.clip(a + b / jnp.tan(angle), min_mass, max_mass)
            return jnp.where(outside, proposal, sample)

        return jax.lax.cond(jnp.any(outside), sample_tail, lambda: sample)

    return relbw


DEFAULT_CONVERTER = {
    "gauss": gauss_factory,
    "bw": breitwigner_factory,
    "relbw": relativistic_breitwigner_factory,
}
