# Usage

`GenParticle` represents a stable or decaying particle and requires a name and mass. The mass
can be an array-like value or a function. A mass function describes a variable mass, such as
that of a resonance. It takes four arguments and returns an array-like object of shape
`(n_events,)`:

- The minimum mass allowed by the decay chain, of shape `(n_events,)`.
- The maximum mass available, of shape `(n_events,)`.
- The number of events to generate.
- A JAX PRNG key.

The mass bounds allow the function to account for thresholds and sample kinematically allowed
values.

!!! note
    `phasespace` raises a `ValueError` if a kinematically forbidden decay is requested.

## A simple example

Use `GenParticle.set_children` to build a decay chain. For example, the following chain describes
$B^{0}\to K^{*}\gamma$ followed by $K^*\to K\pi$, with a fixed $K^*$ mass:

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

Call `generate` with the desired number of events. It returns:

- Normalized event weights as an array of shape `(n_events,)`.
- A dictionary keyed by particle name. Its values are four-momenta represented as arrays of shape
  `(n_events, 4)` or as `vector.Momentum` objects, according to the `as_vectors` argument.

```python
N_EVENTS = 1000

weights, particles = bz.generate(n_events=N_EVENTS, as_vectors=True)
# or
weights, particles = bz.generate(n_events=N_EVENTS)
```

Convert JAX arrays to NumPy arrays with `np.asarray(obj)` when needed.

## Boosting the particles

Particles are generated in the rest frame of the parent at the top of the chain. Pass its
four-momentum through `boost_to` to generate events in another frame. For an example with a
distribution of parent momenta, see `test_kstargamma_kstarnonresonant_lhc` and
`test_k1gamma_kstarnonresonant_lhc` in `tests/test_physics.py`.

`boost_to` accepts a four-momentum array with components `(px, py, pz, energy)` or a
`vector.Momentum` object. An array can have shape `(n_events, 4)` to specify one momentum per
event.

```python
N_EVENTS = 1000

# Generate the top particle with a momentum of 100 GeV
top_momentum = np.array([0, 0, 100, np.sqrt(100**2 + B0_MASS**2)])
# or
top_momentum = vector.array({'px': [0], 'py': [0], 'pz': [100], 'E': [np.sqrt(100**2 + B0_MASS**2)]})
weights, particles = bz.generate(n_events=N_EVENTS, boost_to=top_momentum)
```

## Weights

Set `generate_unnormalized` in `generate` to obtain unnormalized weights. The method then returns
the weights, the per-event maximum weight and the particle dictionary.

Repeated generation can use ordinary Python loops:

```python
for i in range(5):
    weights, particles = bz.generate(n_events=100, key=i)
    # ...
    # (do something with weights and particles)
    # ...
```

## Resonances with variable mass

To sample resonance masses, supply a mass function in place of a fixed value. It receives
`mass(min_mass, max_mass, n_events, key)`: per-event lower and upper mass bounds, the number of
events and a JAX PRNG key. It must return an array-like object of shape `(n_events,)` and be
compatible with `jax.jit`, using
[jax.numpy](https://docs.jax.dev/en/latest/jax.numpy.html) and
[jax.random](https://docs.jax.dev/en/latest/jax.random.html).

Ready-made mass shapes for resonances (Gaussian, Breit-Wigner and relativistic Breit-Wigner) are
available in [`phasespace.fromdecay.mass_functions`](api.md#phasespacefromdecay).

For the preceding example, a Gaussian approximation to the resonance mass gives the following
$B^{0}\to K^{*}\gamma$ chain. See `tests/helpers/decays.py` for more examples.

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

For a simple $n$-body decay, `phasespace.nbody_decay` takes:

- The mass of the top particle.
- A list of daughter masses.
- The name of the top particle (optional).
- The names of the daughter particles (optional).

If names are omitted, the function assigns `top` to the parent and `p_{i}` to the daughters. For
example, to generate $B^0\to K\pi$:

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

Here, `decay` is a `GenParticle` with the specified daughters.

## Eager execution

By default, `phasespace` compiles generation with `jax.jit`. On the first call, JAX traces the
computation with abstract arrays and compiles it. This incurs an initial cost, while later calls
can reuse the compiled computation. Traced values are unavailable to ordinary Python debugging
code inside the function.

The number of events is a *static* argument: a new `n_events` value triggers compilation, whereas
repeated calls with the same value reuse the compiled function. Reusing event counts therefore
avoids additional compilation.

For debugging, use `jax.disable_jit()` or set `PHASESPACE_EAGER=1` to run eagerly.

## Double precision

Phase space generation requires double precision for numerical stability. Near threshold,
catastrophic cancellation in `pdk` degrades relative energy-momentum conservation from about
1e-15 to 1e-6 in single precision. Therefore, `generate` enables JAX's double precision mode
for the duration of the call and returns `float64` arrays. Importing `phasespace` leaves the
global JAX setting unchanged.

When x64 mode is disabled, subsequent JAX operations may downcast the returned arrays to
`float32` and issue a warning. Conversion with `np.asarray` preserves their precision. For
subsequent JAX calculations, enable x64 with `jax.config.update("jax_enable_x64", True)`.

The helpers in [`phasespace.kinematics`][phasespace.kinematics] follow the caller's precision so
they remain compatible with `jax.jit`. Use them with x64 enabled for double precision.

Decays require positive available phase space. At the exact threshold, the phase-space volume and
maximum weight are zero, so generation raises `ValueError`. Boosts support batches containing
both stationary and moving parents. Internal boosts use the known invariant mass directly to
preserve precision for light intermediate systems.

Rotation coefficients are shared across particles through a
[JAX optimization barrier](https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html).
This avoids repeated transcendental evaluations in compiled CPU kernels. The performance benefit
depends on event count, hardware and compiler; random draws and precision are unchanged.

## Running on a GPU

Install a JAX build compatible with your CUDA version, for example:

```bash
pip install "jax[cuda13]"
```

JAX provides device selection for `phasespace`:

```python
import jax

print(jax.devices())  # [CudaDevice(id=0)] once a CUDA-enabled jaxlib is installed

with jax.default_device(jax.devices("gpu")[0]):
    weights, particles = bz.generate(n_events=10_000, key=42)
```

Alternatively, `JAX_PLATFORMS=cuda` or `JAX_PLATFORMS=cpu` selects a backend for the whole run.
This setting restricts JAX to the selected backend: for example, `jax.devices("cpu")` raises
`RuntimeError` under `JAX_PLATFORMS=cuda`. Leave it unset to access both backends in one process.

The computation is primarily limited by memory bandwidth. Measure performance on your device
with `benchmark/bench_phasespace.py`.

CPU and GPU results can differ by a few units in the last place (ULPs) because their arithmetic
differs. The PRNG produces the same draws on both backends.

### Memory

Each event requires 32 bytes per particle for four `float64` components, plus 8 bytes for its
weight. Generating 10 million `B -> 3pi` events returns 0.97 GiB of data and peaks at 3.2 GiB of
device memory. Budget approximately three times the result size. If generation exceeds available
memory, use `chunk_size` to divide it into batches:

```python
weights, particles = bz.generate(n_events=10_000_000, key=42, chunk_size=1_000_000)
```

For the same 10 million events, peak memory falls to 2.1 GiB. Chunking bounds the memory used
during generation, but the chunks and concatenated result coexist at the end. Consequently, peak
memory remains at least about twice the result size, regardless of chunk size. If the result
itself does not fit, consume chunks separately.

Each chunk receives a separate split of `key`. Thus, chunked and unchunked runs with the same
key draw different samples, although both remain reproducible. Within one call, generation uses
at most two event-count specializations: the full chunk and the remainder. Across calls with
different totals, each distinct remainder size can require another compilation.

XLA preallocates 75% of GPU memory on the first computation. On a shared GPU, set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` or `XLA_PYTHON_CLIENT_MEM_FRACTION=.5` to limit allocation.

`PHASESPACE_EAGER=1` disables JIT compilation and dispatches operations individually. It is
useful for CPU debugging but very slow on a GPU.

## Random numbers

JAX random number generation is functional: an explicit key passes through the computation.
Functions that generate random numbers therefore take a `key` argument. It can be:

- `None`: create a new key from OS entropy, so the generation is not reproducible.
- An integer: create a key from that value; reusing the value reproduces the output.
- A JAX PRNG key created by `jax.random.key`: use that key directly.

Passing the same key twice returns the same events.
