# Usage

The base of `phasespace` is the `GenParticle` object. This object, which represents a particle,
either stable or decaying, has only one mandatory argument, its name.

In most cases (except for the top particle of a decay), one wants to also specify its mass, which
can either be array-like, or a function. Functions are used to specify the mass of particles such
as resonances, which are not fixed but vary according to a broad distribution. These mass functions
get four arguments and must return an array-like object of shape `(n_events,)`:

- The minimum mass allowed by the decay chain, of shape `(n_events,)`.
- The maximum mass available, of shape `(n_events,)`.
- The number of events to generate.
- A JAX PRNG key.

This function signature allows to handle threshold effects cleanly, giving enough information to
produce kinematically allowed decays.

!!! note
    `phasespace` raises a `ValueError` if a kinematically forbidden decay is requested.

## A simple example

With these considerations in mind, one can build a decay chain by using the `set_children` method
of the `GenParticle` class. As an example, to build the $B^{0}\to K^{*}\gamma$ decay in which
$K^*\to K\pi$ with a fixed mass, one would write:

```python
from phasespace import GenParticle
import numpy as np
from particle import literals as lp
import vector

B0_MASS = lp.B_0.mass
KSTARZ_MASS = lp.Kst_892_0.mass
PION_MASS = lp.pi_plus.mass
KAON_MASS = lp.K_plus.mass

pion = GenParticle('pi-', PION_MASS)
kaon = GenParticle('K+', KAON_MASS)
kstar = GenParticle('K*', KSTARZ_MASS).set_children(pion, kaon)
gamma = GenParticle('gamma', 0)
bz = GenParticle('B0', B0_MASS).set_children(kstar, gamma)
```

Phase space events can be generated using the `generate` method, which gets the number of events to
generate as input. The method returns:

- The normalized weights of each event, as an array of dimension `(n_events,)`.
- The 4-momenta of the generated particles as values of a dictionary with the particle name as key.
  These momenta are *either* expressed as arrays of dimension `(n_events, 4)` or `vector.Momentum`
  objects, depending on the `as_vectors` flag given to `generate`.

```python
N_EVENTS = 1000

weights, particles = bz.generate(n_events=N_EVENTS, as_vectors=True)
# or
weights, particles = bz.generate(n_events=N_EVENTS)
```

JAX arrays can always be converted to a numpy array (if really needed) through `np.asarray(obj)`.

## Boosting the particles

The particles are generated in the rest frame of the top particle. To produce them at a given
momentum of the top particle, one can pass these momenta with the `boost_to` argument in
`generate`. This latter approach can be useful if the momentum of the top particle is generated
according to some distribution, for example the kinematics of the LHC (see
`test_kstargamma_kstarnonresonant_lhc` and `test_k1gamma_kstarnonresonant_lhc` in
`tests/test_physics.py` to see how this could be done).

The `boost_to` argument can be a 4-momentum array of shape `(n_events, 4)` with
`(px, py, pz, energy)` or a `vector.Momentum` (both a momentum and a Lorentz vector).

```python
N_EVENTS = 1000

# Generate the top particle with a momentum of 100 GeV
top_momentum = np.array([0, 0, 100, np.sqrt(100**2 + B0_MASS**2)])
# or
top_momentum = vector.array({'px': [0], 'py': [0], 'pz': [100], 'E': [np.sqrt(100**2 + B0_MASS**2)]})
weights, particles = bz.generate(n_events=N_EVENTS, boost_to=top_momentum)
```

## Weights

Additionally, it is possible to obtain the unnormalized weights by using the
`generate_unnormalized` flag in `generate`. In this case, the method returns the unnormalized
weights, the per-event maximum weight and the particle dictionary.

Iterative generation can be performed using normal python loops without loss in performance:

```python
for i in range(5):
    weights, particles = bz.generate(n_events=100, key=i)
    # ...
    # (do something with weights and particles)
    # ...
```

## Resonances with variable mass

To generate the mass of a resonance, we need to give a function as its mass instead of a floating
number. This function is called as `mass(min_mass, max_mass, n_events, key)`: the per-event lower
mass allowed, the per-event upper mass allowed, the number of events and a JAX PRNG key. It should
return an array-like object with the generated masses and shape `(n_events,)`, and has to be
jit-compatible, i.e. written with
[jax.numpy](https://docs.jax.dev/en/latest/jax.numpy.html) and
[jax.random](https://docs.jax.dev/en/latest/jax.random.html).

Ready-made mass shapes for resonances (Gaussian, Breit-Wigner and relativistic Breit-Wigner) are
available in [`phasespace.fromdecay.mass_functions`](api.md#phasespacefromdecay).

Following with the same example as above, and approximating the resonance shape by a gaussian, we
could write the $B^{0}\to K^{*}\gamma$ decay chain as (more details can be found in
`tests/helpers/decays.py`):

```python
import jax
from phasespace import numpy as jnp
from phasespace import GenParticle

KSTARZ_MASS = 895.81
KSTARZ_WIDTH = 47.4

def kstar_mass(min_mass, max_mass, n_events, key):
    # a normal distribution truncated to the kinematically allowed range: jax samples the
    # standard normal within the standardized bounds, which is then scaled back
    standard = jax.random.truncated_normal(key,
                                           lower=(min_mass - KSTARZ_MASS) / KSTARZ_WIDTH,
                                           upper=(max_mass - KSTARZ_MASS) / KSTARZ_WIDTH,
                                           shape=(n_events,),
                                           dtype=jnp.float64)
    return KSTARZ_MASS + KSTARZ_WIDTH * standard

bz = GenParticle('B0', B0_MASS).set_children(GenParticle('K*0', mass=kstar_mass)
                                             .set_children(GenParticle('K+', mass=KAON_MASS),
                                                           GenParticle('pi-', mass=PION_MASS)),
                                             GenParticle('gamma', mass=0.0))

bz.generate(n_events=500)
```

## Shortcut for simple decays

The generation of simple $n$-body decay chains can be done using the `nbody_decay` function of
`phasespace`, which takes

- The mass of the top particle.
- The mass of children particles as a list.
- The name of the top particle (optional).
- The names of the children particles (optional).

If the names are not given, `top` and `p_{i}` are assigned. For example, to generate
$B^0\to K\pi$, one would do:

```python
import phasespace
from particle import literals as lp

N_EVENTS = 1000

B0_MASS = lp.B_0.mass
PION_MASS = lp.pi_plus.mass
KAON_MASS = lp.K_plus.mass

decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, KAON_MASS],
                               top_name="B0", names=["pi", "K"])
weights, particles = decay.generate(n_events=N_EVENTS)
```

In this example, `decay` is simply a `GenParticle` with the corresponding children.

## Eager execution

By default, `phasespace` uses JIT (*just-in-time*) compilation with `jax.jit` to greatly speed up
the generation of events. Simplified, this means that the first time a decay is generated, a
symbolic array *without a concrete value* is used and the computation is compiled. As a user
calling the function, you will not notice this, the output will be the same as if the function was
executed eagerly. The consequence is two-fold: on one hand the initial overhead is higher with a
significant speedup for subsequent generations, on the other hand, the values of the generated
particles *inside the function* are not available in pure Python (e.g. for debugging basically).

The number of events is a *static* argument of the compiled function: generating with a new
`n_events` recompiles, while repeated calls with the same value reuse the compiled function. Prefer
therefore to generate repeatedly with the same number of events.

If you need to debug the internals, using `jax.disable_jit` (or the environment variable
`PHASESPACE_EAGER=1`) will make everything run numpy-like.

## Double precision

Importing `phasespace` enables the double precision mode of JAX, which is off by default, via
`jax.config.update("jax_enable_x64", True)`. This is a process-wide setting and required for
correctness: the phase space computation is not numerically stable in single precision.

## Random numbers

Random number generation in JAX is purely functional: rather than relying on a global generator
state, an explicit key is threaded through the computation. Every function that generates random
numbers therefore takes a `key` argument, which can be

- `None`, in which case a new key is created from OS entropy. The generation is then not
  reproducible.
- a number, which is used to create a key. Using the same number again results in the same output.
- a JAX PRNG key as created by `jax.random.key`, which is used directly.

Note that, unlike a stateful generator, passing the *same* key twice returns exactly the same
events.
