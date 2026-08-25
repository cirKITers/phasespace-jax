"""Mass distribution functions for resonant particles.

This module provides factory functions that create mass distribution functions for resonant
particles. Each factory returns a callable with the signature
``(min_mass, max_mass, n_events, key)`` that samples masses truncated to ``[min_mass, max_mass]``
and is usable inside jitted code.

Sampling is done by inverse transform sampling (see e.g. L. Devroye, *Non-Uniform Random Variate
Generation*, Springer 1986, Ch. II), analytically where a closed-form quantile function exists and
on a precomputed grid for the relativistic Breit-Wigner, which has none.
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
        far into the tails.
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

    def relbw(min_mass, max_mass, n_events, key):
        uniform = jax.random.uniform(key, (n_events,), dtype=jnp.float64)
        cdf_low = jnp.interp(min_mass, grid_mass, grid_cdf)
        cdf_high = jnp.interp(max_mass, grid_mass, grid_cdf)
        quantile = cdf_low + uniform * (cdf_high - cdf_low)
        return jnp.interp(quantile, grid_cdf, grid_mass)

    return relbw


DEFAULT_CONVERTER = {
    "gauss": gauss_factory,
    "bw": breitwigner_factory,
    "relbw": relativistic_breitwigner_factory,
}
