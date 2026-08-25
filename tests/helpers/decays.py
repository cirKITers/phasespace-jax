#!/usr/bin/env python3
# =============================================================================
# @file   decays.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   07.03.2019
# =============================================================================
"""Some physics models to test with."""

import jax
import jax.numpy as jnp

from phasespace import GenParticle

# Use RapidSim values (https://github.com/gcowan/RapidSim/blob/master/config/particles.dat)
B0_MASS = 5279.58
PION_MASS = 139.57018
KAON_MASS = 493.677
K1_MASS = 1272.0
K1_WIDTH = 90.0
KSTARZ_MASS = 895.81
KSTARZ_WIDTH = 47.4


def _truncated_normal(loc, scale, min_mass, max_mass, n_events, key):
    """Sample a normal distribution truncated to ``[min_mass, max_mass]``.

    ``jax.random.truncated_normal`` draws from the *standard* normal restricted to the given
    bounds, so the bounds are standardized and the samples shifted back afterwards.
    """
    loc = jnp.asarray(loc, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    min_mass = jnp.asarray(min_mass, dtype=jnp.float64)
    max_mass = jnp.asarray(max_mass, dtype=jnp.float64)
    standard = jax.random.truncated_normal(
        key,
        lower=(min_mass - loc) / scale,
        upper=(max_mass - loc) / scale,
        shape=(n_events,),
        dtype=jnp.float64,
    )
    return loc + scale * standard


def b0_to_kstar_gamma(kstar_width=KSTARZ_WIDTH):
    """Generate B0 -> K*gamma."""

    def kstar_mass(min_mass, max_mass, n_events, key):
        if kstar_width > 0:
            return _truncated_normal(KSTARZ_MASS, kstar_width, min_mass, max_mass, n_events, key)
        return jnp.broadcast_to(jnp.asarray(KSTARZ_MASS, dtype=jnp.float64), (n_events,))

    return GenParticle("B0", B0_MASS).set_children(
        GenParticle("K*0", mass=kstar_mass).set_children(
            GenParticle("K+", mass=KAON_MASS), GenParticle("pi-", mass=PION_MASS)
        ),
        GenParticle("gamma", mass=0.0),
    )


def bp_to_k1_kstar_pi_gamma(k1_width=K1_WIDTH, kstar_width=KSTARZ_WIDTH):
    """Generate B+ -> K1 (-> K* (->K pi) pi) gamma."""

    def res_mass(mass, width, min_mass, max_mass, n_events, key):
        if kstar_width > 0:
            return _truncated_normal(mass, width, min_mass, max_mass, n_events, key)
        return jnp.broadcast_to(jnp.asarray(mass, dtype=jnp.float64), (n_events,))

    def k1_mass(min_mass, max_mass, n_events, key):
        return res_mass(K1_MASS, k1_width, min_mass, max_mass, n_events, key)

    def kstar_mass(min_mass, max_mass, n_events, key):
        return res_mass(KSTARZ_MASS, kstar_width, min_mass, max_mass, n_events, key)

    return GenParticle("B+", B0_MASS).set_children(
        GenParticle("K1+", mass=k1_mass).set_children(
            GenParticle("K*0", mass=kstar_mass).set_children(
                GenParticle("K+", mass=KAON_MASS), GenParticle("pi-", mass=PION_MASS)
            ),
            GenParticle("pi+", mass=PION_MASS),
        ),
        GenParticle("gamma", mass=0.0),
    )


# EOF
