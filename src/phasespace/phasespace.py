#!/usr/bin/env python3
# =============================================================================
# @file   phasespace.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   25.02.2019
# =============================================================================
"""Implementation of the Raubold and Lynch method to generate n-body events.

The code is based on the GENBOD function (W515 from CERNLIB), documented in:

F. James, Monte Carlo Phase Space, CERN 68-15 (1968)
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from math import pi
from typing import TYPE_CHECKING, NoReturn

import jax
import jax.numpy as jnp
import numpy as np

from . import kinematics as kin
from .precision import with_float64
from .random import KeyLike, ensure_key

if TYPE_CHECKING:
    import vector


def process_list_to_tensor(lst):
    """Convert a list to an array.

    The list is converted to an array and transposed to get the proper shape.

    Notes:
        If ``lst`` is an array, nothing is done to it other than convert it to ``float64``.

    Args:
        lst (list): List to convert.

    Returns:
        ``jax.Array``
    """
    if isinstance(lst, list):
        lst = jnp.transpose(jnp.asarray(lst, dtype=jnp.float64))
    return jnp.asarray(lst, dtype=jnp.float64)


def sort_rows(values, n_columns):
    """Sort every row of ``values`` in ascending order, for a small static number of columns.

    Args:
        values (``jax.Array``): Array of shape ``(n_events, n_columns)``.
        n_columns (int): Number of columns, known at trace time.

    Returns:
        ``jax.Array``: ``values`` with each row sorted ascending.

    Notes:
        This is an odd-even transposition network (see e.g. D. E. Knuth, *The Art of Computer
        Programming*, Vol. 3, 2nd ed., §5.3.4) rather than a call to ``jnp.sort``, because the axis
        sorted here is the short one: it holds ``n_particles - 2`` entries while the array has one
        row per event. XLA lowers ``jnp.sort`` to its general sort along that axis, which on GPU
        costs ~860 ms for a ``(1e6, 1)`` array against ~0.16 ms for this network, and dominated the
        whole generation. The number of columns is a Python integer, so the network unrolls into a
        handful of elementwise ``minimum``/``maximum`` passes.

        The result matches ``jnp.sort`` exactly for any input the generation produces, but not for
        subnormals: the CPU backend flushes those to zero in ``jnp.minimum`` and not in ``jnp.sort``.
        This is unreachable here, as the only caller sorts ``jax.random.uniform`` draws, whose
        smallest non-zero value is ~1e-6.
    """
    if n_columns < 2:
        return values
    columns = [values[:, i : i + 1] for i in range(n_columns)]
    # n rounds of compare-exchange on alternating adjacent pairs sort n elements
    for round_number in range(n_columns):
        for i in range(round_number % 2, n_columns - 1, 2):
            lower, upper = columns[i], columns[i + 1]
            columns[i], columns[i + 1] = jnp.minimum(lower, upper), jnp.maximum(lower, upper)
    return jnp.concatenate(columns, axis=1)


def pdk(a, b, c):
    """Calculate the PDK (2-body phase space) function.

    Based on Eq. (9.17) in CERN 68-15 (1968).

    Args:
        a (``jax.Array``): :math:`M_{i+1}` in Eq. (9.17).
        b (``jax.Array``): :math:`M_{i}` in Eq. (9.17).
        c (``jax.Array``): :math:`m_{i+1}` in Eq. (9.17).

    Returns:
        ``jax.Array``
    """
    x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c)
    return jnp.sqrt(x) / (jnp.asarray(2.0, dtype=jnp.float64) * a)


class GenParticle:
    """Representation of a particle.

    Instances of this class can be combined with each other to build decay chains,
    which can then be used to generate phase space events through the ``generate``
    method.

    A ``GenParticle`` must have
        - a ``name``, which is ensured not to clash with any others in
            the decay chain.
        - a ``mass``, which can be either a number or a function to generate it according to
            a certain distribution. The returned ``jax.Array`` needs to have shape ``(nevents,)``.
            In this case, the particle is not considered as having a
            fixed mass and the ``has_fixed_mass`` method will return False.

    It may also have:

        - Children, ie, decay products, which are also ``GenParticle`` instances.


    Args:
        name (str): Name of the particle.
        mass (float, array-like, callable): Mass of the particle. If it's a float, it gets
            converted to array-like. If it is a callable, it is called as
            ``mass(min_mass, max_mass, n_events, key)`` and has to be jit-compatible.
    """

    @with_float64
    def __init__(self, name: str, mass: Callable | int | float | np.typing.ArrayLike) -> None:
        self.name = name
        self.children = []
        self._mass_val = mass
        if callable(mass):

            @functools.wraps(mass)
            def mass_preprocessed(*args, mass=mass, **kwargs):
                return jnp.atleast_1d(
                    jnp.asarray(mass(*args, **kwargs), dtype=jnp.float64)  # ty: ignore[call-top-callable]
                )
        else:
            mass_preprocessed = jnp.atleast_1d(jnp.asarray(mass, dtype=jnp.float64))

        self._mass = mass_preprocessed
        self._generate_called = False  # not yet called, children can be set
        self._jitted_recursive_generate = None

    def __repr__(self):
        return "<phasespace.GenParticle: name='{}' mass={} children=[{}]>".format(
            self.name,
            f"{self._mass_val:.2f}" if self.has_fixed_mass else "variable",
            ", ".join(child.name for child in self.children),
        )

    def _do_names_clash(self, particles):
        def get_list_of_names(part):
            output = [part.name]
            for child in part.children:
                output.extend(get_list_of_names(child))
            return output

        names_to_check = [self.name]
        for part in particles:
            names_to_check.extend(get_list_of_names(part))
        # Find top
        dup_names = {name for name in names_to_check if names_to_check.count(name) > 1}
        if dup_names:
            return dup_names
        return None

    @with_float64
    def get_mass(
        self,
        min_mass: jax.Array | None = None,
        max_mass: jax.Array | None = None,
        n_events: int | None = None,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Get the particle mass.

        If the particle is resonant, the mass function is called as
        ``mass(min_mass, max_mass, n_events, key)``.

        Args:
            min_mass (array): Lower mass range. Defaults to None, which
                is only valid in the case of fixed mass.
            max_mass (array): Upper mass range. Defaults to None, which
                is only valid in the case of fixed mass.
            n_events (int): Number of events to produce. Has to be specified if the particle is resonant.
            key (``jax.Array``): JAX PRNG key, handed to the mass function. Has to be specified if the
                particle is resonant.

        Returns:
            ``jax.Array``: Mass of the particles, either a scalar or shape ``(nevents,)``

        Raises:
            ValueError: If the mass is requested and has not been set.
        """
        if self.has_fixed_mass:
            return self._mass  # ty: ignore[invalid-return-type]
        if key is None:
            raise ValueError(f"A key is needed to generate the mass of the resonance '{self.name}'.")
        min_mass = jnp.reshape(min_mass, (n_events,))  # ty: ignore[invalid-argument-type]
        max_mass = jnp.reshape(max_mass, (n_events,))  # ty: ignore[invalid-argument-type]
        return self._mass(min_mass, max_mass, n_events, key)  # ty: ignore[call-non-callable]

    @property
    def has_fixed_mass(self):
        """bool: Is the mass a callable function?"""
        return not callable(self._mass)

    def set_children(self, *children):
        """Assign children.

        Args:
            children (GenParticle): Two or more children to assign to the current particle.

        Returns:
            self

        Raises:
            ValueError: If there is an inconsistency in the parent/children relationship, ie,
            if children were already set, if their parent was or if less than two children were given.
            KeyError: If there is a particle name clash.
            RuntimeError: If ``generate`` was already called before.
        """
        if self._generate_called:
            raise RuntimeError("Cannot set children after the first call to `generate`.")
        if self.children:
            raise ValueError("Children already set!")
        if len(children) <= 1:
            raise ValueError(f"Have to set at least 2 children, not {len(children)} for a particle to decay")
        # Check name clashes
        if name_clash := self._do_names_clash(children):
            raise KeyError(f"Particle name {name_clash} already used")
        self.children = children
        return self

    @property
    def has_children(self):
        """bool: Does the particle have children?"""
        return bool(self.children)

    @property
    def has_grandchildren(self):
        """bool: Does the particle have grandchildren?"""
        if not self.children:
            return False
        return any(child.has_children for child in self.children)

    @staticmethod
    def _preprocess(momentum, n_events):
        """Preprocess momentum input and determine number of events to generate.

        Args:
            momentum: Momentum vector, of shape ``(x, 4)``, where x is optional.
            n_events: Number of events to generate. If ``None``, the number of events
            to generate is calculated from the shape of ``momentum``.

        Returns:
            tuple: Processed ``momentum`` and ``n_events``.
        """
        momentum = process_list_to_tensor(momentum)

        # Check sanity of inputs
        if len(momentum.shape) not in (1, 2):
            raise ValueError(f"Bad shape for momentum -> {list(momentum.shape)}")

        if n_events is None:
            n_events = momentum.shape[0] if len(momentum.shape) == 2 else 1
        n_events = int(n_events)
        # Now preparation of tensors
        if len(momentum.shape) == 1:
            momentum = jnp.expand_dims(momentum, axis=0)
        return momentum, n_events

    @staticmethod
    def _get_w_max(available_mass, masses):
        emmax = available_mass + masses[:, 0:1]
        emmin = jnp.zeros_like(emmax, dtype=jnp.float64)
        w_max = jnp.ones_like(emmax, dtype=jnp.float64)
        for i in range(1, masses.shape[1]):
            emmin += masses[:, i - 1 : i]
            emmax += masses[:, i : i + 1]
            w_max *= pdk(emmax, emmin, masses[:, i : i + 1])
        return w_max

    def _generate(self, momentum, n_events, key):
        """Generate an n-body decay according to the Raubold and Lynch method.

        The number and mass of the children particles are taken from self.children.

        Notes:
            This method generates the same results as the GENBOD routine.

        Args:
            momentum (array): Momentum of the parent particle. All generated particles
                will be boosted to that momentum.
            n_events (int): Number of events to generate.
            key: JAX PRNG key.

        Returns:
            tuple: Result of the generation (per-event weights, maximum weights, output particles,
                their output masses and whether the decay is kinematically allowed).
        """
        self._generate_called = True
        if not self.children:
            raise ValueError("No children have been configured")
        p_top, n_events = self._preprocess(momentum, n_events)
        top_mass = jnp.broadcast_to(kin.mass(p_top), (n_events, 1))
        n_particles = len(self.children)

        # Prepare masses
        def recurse_stable(part):
            output_mass = jnp.zeros((1,), dtype=jnp.float64)
            for child in part.children:
                if child.has_fixed_mass:
                    output_mass += child.get_mass()
                else:
                    output_mass += recurse_stable(child)
            return output_mass

        mass_from_stable = jnp.broadcast_to(
            sum(
                (child.get_mass() for child in self.children if child.has_fixed_mass),
                jnp.zeros((1,), dtype=jnp.float64),
            ),
            (n_events, 1),
        )
        max_mass = top_mass - mass_from_stable
        masses = []
        for child in self.children:
            if child.has_fixed_mass:
                masses.append(jnp.broadcast_to(child.get_mass(), (n_events, 1)))
            else:
                # Recurse that particle to know the minimum mass we need to generate
                min_mass = jnp.broadcast_to(recurse_stable(child), (n_events, 1))
                key, mass_key = jax.random.split(key)
                mass = child.get_mass(min_mass, max_mass, n_events, mass_key)
                mass = jnp.reshape(mass, (n_events, 1))
                max_mass -= mass
                masses.append(mass)
        masses = jnp.concatenate(masses, axis=-1)
        available_mass = top_mass - jnp.sum(masses, axis=1, keepdims=True)
        # Kinematically forbidden decays cannot raise from inside a jitted function; the flag is
        # propagated up and checked eagerly in `generate`.
        allowed = jnp.all(available_mass > 0.0)
        # Calculate the max weight, initial beta, etc
        w_max = self._get_w_max(available_mass, masses)
        p_top_boost = kin.boost_components(p_top)
        # Start the generation
        key, uniform_key, part2_key = jax.random.split(key, 3)
        random_numbers = jax.random.uniform(uniform_key, (n_events, n_particles - 2), dtype=jnp.float64)
        random = jnp.concatenate(
            [
                jnp.zeros((n_events, 1), dtype=jnp.float64),
                sort_rows(random_numbers, n_particles - 2),
                jnp.ones((n_events, 1), dtype=jnp.float64),
            ],
            axis=1,
        )
        sum_ = jnp.zeros((n_events, 1), dtype=jnp.float64)
        inv_masses = []
        # TODO(Mayou36): rewrite with cumsum?
        for i in range(n_particles):
            sum_ += masses[:, i : i + 1]
            inv_masses.append(random[:, i : i + 1] * available_mass + sum_)
        generated_particles, weights = self._generate_part2(inv_masses, masses, n_events, n_particles, key=part2_key)
        # Final boost of all particles
        generated_particles = [kin.lorentz_boost(part, p_top_boost) for part in generated_particles]
        return (
            jnp.reshape(weights, (n_events,)),
            jnp.reshape(w_max, (n_events,)),
            generated_particles,
            masses,
            allowed,
        )

    @staticmethod
    def _generate_part2(inv_masses, masses, n_events, n_particles, key):
        pds = []
        # Calculate weights of the events
        for i in range(n_particles - 1):
            pds.append(
                pdk(
                    inv_masses[i + 1],
                    inv_masses[i],
                    masses[:, i + 1 : i + 2],
                )
            )
        weights = jnp.prod(jnp.stack(pds), axis=0)
        zero_component = jnp.zeros_like(pds[0], dtype=jnp.float64)
        generated_particles = [
            jnp.concatenate(
                [
                    zero_component,
                    pds[0],
                    zero_component,
                    jnp.sqrt(jnp.square(pds[0]) + jnp.square(masses[:, 0:1])),
                ],
                axis=1,
            )
        ]
        part_num = 1
        while True:
            generated_particles.append(
                jnp.concatenate(
                    [
                        zero_component,
                        -pds[part_num - 1],
                        zero_component,
                        jnp.sqrt(jnp.square(pds[part_num - 1]) + jnp.square(masses[:, part_num : part_num + 1])),
                    ],
                    axis=1,
                )
            )

            key, cos_z_key, ang_y_key = jax.random.split(key, 3)
            cos_z = jnp.asarray(2.0, dtype=jnp.float64) * jax.random.uniform(
                cos_z_key, (n_events, 1), dtype=jnp.float64
            ) - jnp.asarray(1.0, dtype=jnp.float64)
            sin_z = jnp.sqrt(jnp.asarray(1.0, dtype=jnp.float64) - cos_z * cos_z)
            ang_y = (
                jnp.asarray(2.0, dtype=jnp.float64)
                * jnp.asarray(pi, dtype=jnp.float64)
                * jax.random.uniform(ang_y_key, (n_events, 1), dtype=jnp.float64)
            )
            cos_y = jnp.cos(ang_y)
            sin_y = jnp.sin(ang_y)
            # Materialize the shared rotation coefficients once per stage. Otherwise CPU fusion
            # can repeat their transcendental evaluations for each outgoing momentum component.
            cos_z, sin_z, cos_y, sin_y = jax.lax.optimization_barrier((cos_z, sin_z, cos_y, sin_y))
            # Do the rotations
            for j in range(part_num + 1):
                px = kin.x_component(generated_particles[j])
                py = kin.y_component(generated_particles[j])
                # Rotate about z
                generated_particles[j] = jnp.concatenate(
                    [
                        cos_z * px - sin_z * py,
                        sin_z * px + cos_z * py,
                        kin.z_component(generated_particles[j]),
                        kin.time_component(generated_particles[j]),
                    ],
                    axis=1,
                )
                # Rotate about y
                px = kin.x_component(generated_particles[j])
                pz = kin.z_component(generated_particles[j])
                generated_particles[j] = jnp.concatenate(
                    [
                        cos_y * px - sin_y * pz,
                        kin.y_component(generated_particles[j]),
                        sin_y * px + cos_y * pz,
                        kin.time_component(generated_particles[j]),
                    ],
                    axis=1,
                )
            if part_num == (n_particles - 1):
                break
            # The boost is along y. Use gamma = E/M and gamma*beta = p/M directly:
            # recovering gamma from 1 - beta**2 loses precision for light intermediate systems.
            gamma = jnp.sqrt(jnp.square(pds[part_num]) + jnp.square(inv_masses[part_num])) / inv_masses[part_num]
            gamma_beta = pds[part_num] / inv_masses[part_num]
            generated_particles = [
                jnp.concatenate(
                    [
                        kin.x_component(part),
                        gamma * kin.y_component(part) + gamma_beta * kin.time_component(part),
                        kin.z_component(part),
                        gamma * kin.time_component(part) + gamma_beta * kin.y_component(part),
                    ],
                    axis=1,
                )
                for part in generated_particles
            ]
            part_num += 1
        return generated_particles, weights

    def _recursive_generate(
        self,
        n_events: int,
        boost_to=None,
        recalculate_max_weights: bool = False,
        key: jax.Array | None = None,
    ):
        """Recursively generate normalized n-body phase space.

        Events are generated in the rest frame of the particle, unless ``boost_to`` is given.

        Notes:
            In this method, the event weights are returned normalized to their maximum.

        Args:
            n_events (int): Number of events to generate.
            boost_to (array, optional): Momentum vector of shape ``(x, 4)``, where x is optional, to where
                the resulting events will be boosted. If not specified, events are generated
                in the rest frame of the particle.
            recalculate_max_weights (bool, optional): Recalculate the maximum weight of the event
                using all the particles of the tree? This is necessary for the top particle of a decay,
                otherwise the maximum weight calculation is going to be wrong (particles from subdecays
                would not be taken into account). Defaults to False.
            key (``jax.Array``): JAX PRNG key.

        Returns:
            tuple: Result of the generation (per-event weights, maximum weights, output particles,
                their output masses and whether the decay is kinematically allowed).

        Raises:
            ValueError: If a resonance is used as the top particle.
        """
        if boost_to is not None:
            momentum = boost_to
        elif self.has_fixed_mass:
            zero = jnp.zeros((), dtype=jnp.float64)
            momentum = jnp.broadcast_to(
                jnp.stack((zero, zero, zero, self.get_mass()[0]), axis=-1),
                (n_events, 4),
            )
        else:
            raise ValueError("Cannot use resonance as top particle")
        keys = jax.random.split(key, 1 + len(self.children))  # ty: ignore[invalid-argument-type]
        weights, weights_max, parts, children_masses, allowed = self._generate(momentum, n_events, key=keys[0])
        output_particles = {child.name: parts[child_num] for child_num, child in enumerate(self.children)}
        output_masses = {
            child.name: children_masses[:, child_num : child_num + 1] for child_num, child in enumerate(self.children)
        }
        for child_num, child in enumerate(self.children):
            if child.has_children:
                (
                    child_weights,
                    _,
                    child_gen_particles,
                    child_masses,
                    child_allowed,
                ) = child._recursive_generate(
                    n_events=n_events,
                    boost_to=parts[child_num],
                    recalculate_max_weights=False,
                    key=keys[child_num + 1],
                )
                weights *= child_weights
                allowed = jnp.logical_and(allowed, child_allowed)
                output_particles.update(child_gen_particles)
                output_masses.update(child_masses)
        if recalculate_max_weights:

            def build_mass_tree(particle, leaf):
                if particle.has_children:
                    leaf[particle.name] = {}
                    for child in particle.children:
                        build_mass_tree(child, leaf[particle.name])
                else:
                    leaf[particle.name] = output_masses[particle.name]

            def get_flattened_values(dict_):
                output = []
                for val in dict_.values():
                    if isinstance(val, dict):
                        output.extend(get_flattened_values(val))
                    else:
                        output.append(val)
                return output

            def recurse_w_max(parent_mass, current_mass_tree):
                available_mass = parent_mass - sum(get_flattened_values(current_mass_tree))
                masses = []
                w_max = jnp.ones_like(available_mass)
                for child, child_mass in current_mass_tree.items():
                    if isinstance(child_mass, dict):
                        w_max *= recurse_w_max(
                            parent_mass
                            - sum(
                                get_flattened_values(
                                    {ch_it: ch_m_it for ch_it, ch_m_it in current_mass_tree.items() if ch_it != child}
                                )
                            ),
                            child_mass,
                        )
                        masses.append(sum(get_flattened_values(child_mass)))
                    else:
                        masses.append(child_mass)
                masses = jnp.concatenate(masses, axis=1)
                w_max *= self._get_w_max(available_mass, masses)
                return w_max

            mass_tree = {}
            build_mass_tree(self, mass_tree)
            momentum = process_list_to_tensor(momentum)
            if len(momentum.shape) == 1:
                momentum = jnp.expand_dims(momentum, axis=-1)
            weights_max = jnp.reshape(recurse_w_max(kin.mass(momentum), mass_tree[self.name]), (n_events,))
        return weights, weights_max, output_particles, output_masses, allowed

    def _generate_chunked(self, n_events, chunk_size, boost_to, key):
        """Generate ``n_events`` events in chunks of at most ``chunk_size`` and stitch them together.

        Every chunk is a full ``generate`` call of its own, which keeps the input validation and the
        forbidden-decay check in one place. Weights are left unnormalized here because the caller
        normalizes the stitched result: ``weights_max`` is a per-event quantity, so normalizing per
        chunk and concatenating would give the very same numbers.

        Args:
            n_events (int): Total number of events to generate.
            chunk_size (int): Maximum number of events per chunk.
            boost_to: Momentum vector to boost to, already preprocessed, or None.
            key: JAX PRNG key, split once per chunk.

        Returns:
            tuple: The unnormalized event weights, the maximum per-event weights and the momenta of
                the generated particles, each covering all ``n_events`` events.
        """
        sizes = [chunk_size] * (n_events // chunk_size)
        if remainder := n_events % chunk_size:
            sizes.append(remainder)
        keys = jax.random.split(key, len(sizes))
        weights, weights_max, parts = [], [], []
        start = 0
        for size, chunk_key in zip(sizes, keys):
            chunk_boost = boost_to
            if boost_to is not None and boost_to.shape[0] != 1:
                chunk_boost = boost_to[start : start + size]
            chunk = self.generate(size, boost_to=chunk_boost, normalize_weights=False, key=chunk_key)
            weights.append(chunk[0])
            weights_max.append(chunk[1])
            parts.append(chunk[2])
            start += size
        return (
            jnp.concatenate(weights),
            jnp.concatenate(weights_max),
            {name: jnp.concatenate([part[name] for part in parts]) for name in parts[0]},
        )

    @with_float64
    def generate(
        self,
        n_events: int,
        boost_to: jax.Array | vector.Momentum | None = None,
        normalize_weights: bool = True,
        key: KeyLike = None,
        *,
        as_vectors: bool | None = None,
        chunk_size: int | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]] | tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        """Generate normalized n-body phase space as JAX arrays.

        Events are generated in the rest frame of the particle, unless ``boost_to`` is given.

        Notes:
            In this method, the event weights are returned normalized to their maximum.

            The generation is jit-compiled with ``n_events`` as a static argument: calling this
            with a new value of ``n_events`` triggers a recompilation, while repeated calls with
            the same value reuse the compiled function.

            Chunking changes which events are drawn, as every chunk consumes its own split of
            ``key``: ``generate(n, key=k)`` and ``generate(n, key=k, chunk_size=c)`` give different
            but equally valid samples, each of them reproducible. It bounds the memory of the
            generation itself, not of the returned arrays. Each call uses at most two event-count
            specializations (a full chunk and its remainder); different remainder sizes across
            calls can require additional compilations.

        Args:
            n_events (int): Number of events to generate.
            boost_to (optional): Momentum vector of shape ``(x, 4)``, where x is optional, to where
                the resulting events will be boosted in the (px, py, pz, E) format.
                Can also be a ``vector`` momentum Lorentz vector.
                If not specified, events are generated in the rest frame of the particle.
            normalize_weights (bool, optional): Normalize the event weight to its max?
            key (``KeyLike``): Either an integer seed, a JAX PRNG key or None, in which case a
                new key is created from OS entropy (and the generation is not reproducible).
            as_vectors (bool, optional): If True, the output momenta are returned as ``vector`` objects.
            chunk_size (int, optional): Generate the events in chunks of at most this many rather
                than all at once, which bounds the peak memory of the generation. Defaults to None,
                which generates everything in one go.

        Returns:
            tuple: Result of the generation, which varies with the value of ``normalize_weights``:

                - If True, the tuple elements are the normalized event weights as an array of shape
                  ``(n_events,)``, and the momenta of the generated particles as a dictionary of arrays
                  of shape ``(n_events, 4)`` with particle names as keys.

                - If False, the tuple elements are the unnormalized event weights as an array of shape
                  ``(n_events,)``, the maximum per-event weights as an array of shape ``(n_events,)`` and
                  the momenta of the generated particles as a dictionary of arrays of shape
                  ``(n_events, 4)`` with particle names as keys.

        Raises:
            ValueError: If the decay has no positive available phase space, if ``n_events`` and the size of
                ``boost_to`` don't match or if ``chunk_size`` is not positive.
        """
        key = ensure_key(key)
        n_events = int(n_events)
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError(f"chunk_size has to be a positive number of events, not {chunk_size}.")
        if boost_to is not None:
            try:
                import vector
            except ImportError as error:
                _raise_missing_vector_package(error)
            if isinstance(boost_to, vector.Vector):
                if not (isinstance(boost_to, vector.Momentum) and isinstance(boost_to, vector.Lorentz)):
                    raise ValueError("boost_to has to be a momentum Lorentz vector.")
                try:
                    momentum = boost_to.to_pxpypzenergy()
                except Exception as error:
                    raise ValueError(
                        "boost_to has to be a momentum Lorentz vector, failed to convert to pxpypzenergy."
                    ) from error
                boost_to = jnp.stack(
                    [
                        jnp.asarray(momentum.px, dtype=jnp.float64),  # ty: ignore[unresolved-attribute]
                        jnp.asarray(momentum.py, dtype=jnp.float64),  # ty: ignore[unresolved-attribute]
                        jnp.asarray(momentum.pz, dtype=jnp.float64),  # ty: ignore[unresolved-attribute]
                        jnp.asarray(momentum.energy, dtype=jnp.float64),  # ty: ignore[unresolved-attribute]
                    ],
                    axis=-1,
                )
            boost_to = process_list_to_tensor(boost_to)
            if boost_to.shape[0] not in (n_events, 1):
                raise ValueError(
                    f"The number of events requested ({n_events}) doesn't match the boost_to input size "
                    f"of {boost_to.shape}"
                )
        if chunk_size is not None and int(chunk_size) < n_events:
            weights, weights_max, parts = self._generate_chunked(n_events, int(chunk_size), boost_to, key)
        else:
            if self._jitted_recursive_generate is None:
                self._jitted_recursive_generate = jax.jit(
                    self._recursive_generate,
                    static_argnames=("n_events", "recalculate_max_weights"),
                )
            weights, weights_max, parts, _, allowed = self._jitted_recursive_generate(
                n_events=n_events,
                boost_to=boost_to,
                recalculate_max_weights=self.has_grandchildren,
                key=key,
            )
            if not bool(allowed):
                raise ValueError("Forbidden decay: no positive available phase space")
        parts = to_vectors(parts) if as_vectors else parts
        # `parts` holds vector.Momentum objects when as_vectors is set
        if normalize_weights:
            return weights / weights_max, parts  # ty: ignore[invalid-return-type]
        return weights, weights_max, parts  # ty: ignore[invalid-return-type]


def nbody_decay(mass_top: float, masses: list, top_name: str = "", names: list | None = None):
    """Shortcut to build an n-body decay of a GenParticle.

    If the particle names are not given, the top particle is called 'top' and the
    children 'p_{i}', where i corresponds to their position in the ``masses`` sequence.

    Args:
        mass_top (array, list): Mass of the top particle. Can be a list of 4-vectors.
        masses (list): Masses of the child particles.
        top_name (str, optional): Name of the top particle. If not given, the top particle is
            named top.
        names (list, optional): Names of the child particles. If not given, they are build as
            'p_{i}', where i is given by their ordering in the ``masses`` list.

    Returns:
        ``GenParticle``: Particle decay.

    Raises:
        ValueError: If the length of ``masses`` and ``names`` doesn't match.
    """
    if not top_name:
        top_name = "top"
    if not names:
        names = [f"p_{num}" for num in range(len(masses))]
    if len(names) != len(masses):
        raise ValueError("Mismatch in length between children masses and their names.")
    return GenParticle(top_name, mass=mass_top).set_children(
        *(GenParticle(names[num], mass=mass) for num, mass in enumerate(masses))
    )


def to_vectors(particles: dict[str, jax.Array]) -> dict[str, vector.Momentum]:
    """Convert a dictionary of particles to a dictionary of ``vector.Momentum`` instances.

    Args:
        particles (dict): Dictionary of particles, with the keys being the particle names and the
            values being the momenta.

    Returns:
        dict: Dictionary of ``vector.Momentum`` instances with numpy arrays
    """
    try:
        import vector
    except ImportError as error:
        _raise_missing_vector_package(error)
    newparticles = {}
    for name, particle in particles.items():
        px, py, pz, e = np.moveaxis(np.asarray(particle), -1, 0)  # numpy "unstack"
        newparticles[name] = vector.array({"px": px, "py": py, "pz": pz, "energy": e})
    return newparticles  # ty: ignore[invalid-return-type]


def _raise_missing_vector_package(exception: ImportError) -> NoReturn:
    raise ImportError("To use `boost_to`, the `vector` package has to be installed.") from exception
