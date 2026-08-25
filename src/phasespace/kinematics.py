#!/usr/bin/env python3
# =============================================================================
# @file   kinematics.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   12.02.2019
# =============================================================================
"""Basic kinematics."""

import jax.numpy as jnp


def scalar_product(vec1, vec2):
    """Calculate scalar product of two 3-vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Scalar product of the two vectors.
    """
    return jnp.sum(vec1 * vec2, axis=1)


def spatial_component(vector):
    """Extract spatial components of the input Lorentz vector.

    Args:
        vector: Input Lorentz vector (where indexes 0-2 are space, index 3 is time).

    Returns:
        Spatial components (3-vector) of the input Lorentz vector.
    """
    return vector[..., 0:3]


def time_component(vector):
    """Extract time component of the input Lorentz vector.

    Args:
        vector: Input Lorentz vector (where indexes 0-2 are space, index 3 is time).

    Returns:
        Time component of the input Lorentz vector.
    """
    return vector[..., 3:4]


def x_component(vector):
    """Extract spatial X component of the input Lorentz or 3-vector.

    Args:
        vector: Input vector.

    Returns:
        X component of the input vector.
    """
    return vector[..., 0:1]


def y_component(vector):
    """Extract spatial Y component of the input Lorentz or 3-vector.

    Args:
        vector: Input vector.

    Returns:
        Y component of the input vector.
    """
    return vector[..., 1:2]


def z_component(vector):
    """Extract spatial Z component of the input Lorentz or 3-vector.

    Args:
        vector: Input vector.

    Returns:
        Z component of the input vector.
    """
    return vector[..., 2:3]


def mass(vector):
    """Calculate mass scalar for Lorentz 4-momentum.

    Args:
        vector: Input Lorentz momentum vector.

    Returns:
        Mass of the Lorentz 4-momentum vector.
    """
    return jnp.sqrt(jnp.sum(jnp.square(vector) * metric_tensor(), axis=-1, keepdims=True))


def lorentz_vector(space, time):
    """Make a Lorentz vector from spatial and time components.

    Args:
        space: 3-vector of spatial components.
        time: Time component.

    Returns:
        Lorentz 4-vector combining spatial and time components.
    """
    return jnp.concatenate([space, time], axis=-1)


def lorentz_boost(vector, boostvector):
    """Perform Lorentz boost.

    Args:
        vector: 4-vector to be boosted
        boostvector: Boost vector. Can be either 3-vector or 4-vector, since
            only spatial components are used.

    Returns:
        Boosted 4-vector.
    """
    boost = spatial_component(boostvector)
    b2 = jnp.expand_dims(scalar_product(boost, boost), axis=-1)

    def boost_fn():
        gamma = 1.0 / jnp.sqrt(1.0 - b2)
        gamma2 = (gamma - 1.0) / b2
        ve = time_component(vector)
        vp = spatial_component(vector)
        bp = jnp.expand_dims(scalar_product(vp, boost), axis=-1)
        vp2 = vp + (gamma2 * bp + gamma * ve) * boost
        ve2 = gamma * (ve + bp)
        return lorentz_vector(vp2, ve2)

    # if boost vector is zero, return the original vector
    # NOTE: both branches are always evaluated and boost_fn() divides by b2, so the discarded
    # branch holds NaNs for a zero boost. Harmless for values, but `jax.grad` would propagate
    # them; a double-where would be needed if gradients are ever wanted here.
    all_b2_zero = jnp.all(jnp.equal(b2, jnp.zeros_like(b2)))
    return jnp.where(all_b2_zero, vector, boost_fn())


def beta(vector):
    """Calculate beta of a given 4-vector.

    Args:
        vector: Input Lorentz momentum vector.

    Returns:
        Beta (v/c) of the Lorentz momentum vector.
    """
    return mass(vector) / time_component(vector)


def boost_components(vector):
    """Get the boost components of a given 4-vector.

    Args:
        vector: Input Lorentz momentum vector.

    Returns:
        Boost components (3-vector) of the Lorentz momentum vector.
    """
    return spatial_component(vector) / time_component(vector)


def metric_tensor():
    """Metric tensor for Lorentz space (constant).

    Returns:
        Metric tensor for Lorentz space with signature (-1, -1, -1, 1).
    """
    return jnp.asarray([-1.0, -1.0, -1.0, 1.0], dtype=jnp.float64)


# EOF
