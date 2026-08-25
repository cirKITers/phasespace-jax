# PhaseSpace

This repo is a fork of the [original phasespace](https://github.com/zfit/phasespace) repo.
Please refer to their repo for any background information or when citing in scientific publication.

## What is different

The fork replaces the [TensorFlow](https://github.com/tensorflow/tensorflow) dependency with
[JAX](https://github.com/jax-ml/jax) and makes the generation jit-compatible.
The algorithm (GENBOD, Raubold-Lynch, CERN 68-15) are is the exact same as in the original implementation and produce bit-identical events.
We also left the API (and return values) and the `DecayLanguage` integration the same, so you can use `phasespace-jax` it almost as an drop-in replacement.
Please see the [documentation]() for reference on the exact details.

Now, what is different:

| Aspect | [zfit/phasespace](https://github.com/zfit/phasespace) | this fork |
|---|---|---|
| Backend | TensorFlow | JAX |
| Compilation | `tf.function` | `jax.jit`, with `n_events` as a static argument |
| Random numbers | `seed=`, stateful `tf.random.Generator` | `key=`, functional JAX PRNG key
| `n_events` | `int`, `tf.Tensor` or `tf.Variable` | Python `int`, a new value recompiles |
| Mass functions | `f(min_mass, max_mass, n_events[, seed])`, TFP or zfit PDFs | `f(min_mass, max_mass, n_events, key)`, has to be jit-compatible |
| Resonance shapes in `fromdecay` | zfit / zfit-physics PDFs | sampled directly in JAX, no zfit dependency |
| Forbidden decays | `tf.errors.InvalidArgumentError` | `ValueError` |
| `phasespace.numpy` | `tensorflow.experimental.numpy` | `jax.numpy` |
| Distribution name | `phasespace` | `phasespace-jax`, imported as `phasespace` |
| Speed | reference | ≈4-5x faster for 1M `B -> 3pi` events on CPU |

## Installing

To install with pip:

```console
$ pip install phasespace-jax
```

To install the necessary dependencies to be used with
[DecayLanguage](https://github.com/scikit-hep/decaylanguage), use

```console
$ pip install "phasespace-jax[fromdecay]"
```

## How to use

Phasespace can directly be used to generate from a DecayChain using the
[DecayLanguage](https://github.com/scikit-hep/decaylanguage) package as
[explained in the tutorial](https://phasespace.readthedocs.io/en/latest/GenMultiDecay_Tutorial.html).

The generation of simple `n`-body decays can be done using the `nbody_decay` shortcut to create a
decay chain with a very simple interface: one needs to pass the mass of the top particle and the
masses of the children particles as a list, optionally giving the names of the particles. Then, the
`generate` method can be used to produce the desired sample.
For example, to generate $B^0\to K\pi$, we would do:

```python
import phasespace

B0_MASS = 5279.65
PION_MASS = 139.57018
KAON_MASS = 493.677

weights, particles = phasespace.nbody_decay(
    B0_MASS, [PION_MASS, KAON_MASS]
).generate(n_events=1000)
```

The `generate` function returns a `jax.Array` of 1000 elements in the case of `weights` and a dict
of `n particles` (2) arrays of `(1000, 4)` shape, where each of the 4 dimensions corresponds to one
of the components of the generated Lorentz 4-vector. JAX arrays convert to numpy arrays with
`np.asarray(...)`.
All particles are generated in the rest frame of the top particle; boosting to a certain momentum
(or list of momenta) can be achieved by passing the momenta to the `boost_to` argument.

Sequential decays can be handled with the `GenParticle` class (used internally by `generate`) and
its `set_children` method. As an example, to build the $B^{0}\to K^{*}\gamma$ decay in which
$K^*\to K\pi$, we would write:

```python
from phasespace import GenParticle

B0_MASS = 5279.65
KSTARZ_MASS = 895.55
PION_MASS = 139.57018
KAON_MASS = 493.677

kaon = GenParticle('K+', KAON_MASS)
pion = GenParticle('pi-', PION_MASS)
kstar = GenParticle('K*', KSTARZ_MASS).set_children(kaon, pion)
gamma = GenParticle('gamma', 0)
bz = GenParticle('B0', B0_MASS).set_children(kstar, gamma)

weights, particles = bz.generate(n_events=1000)
```

Where we have used the fact that `set_children` returns the parent particle.
In this case, `particles` is a `dict` with the particle names as keys:

```pycon
>>> particles
{'K*': array([[2047.68762461, 1541.15862236,  -72.4256177 , 2715.77793272],
              [1469.35084777, -400.29068127, 2062.57495234, 2715.77793272],
              ...]),
 'K+': array([[1726.88981662,  960.17876701,  -50.47103598, 2037.24225589],
              [ 934.90807089, -454.07101908, 1033.69378006, 1546.7622321 ],
              ...]),
 'gamma': array([[-2047.68762461, -1541.15862236,    72.4256177 ,  2563.87206728],
                 [-1469.35084777,   400.29068127, -2062.57495234,  2563.87206728],
                 ...]),
 'pi-': array([[ 320.79780799,  580.97985535,  -21.95458172,  678.53567684],
               [ 534.44277688,   53.78033781, 1028.88117228, 1169.01570063],
               ...])}
```

### Reproducibility

JAX random number generation is purely functional: instead of a global generator state, an explicit
key is passed in. `generate` accepts an integer seed, a `jax.random` key, or `None`:

```python
import jax

weights, particles = bz.generate(n_events=1000, key=42)          # reproducible
weights, particles = bz.generate(n_events=1000, key=jax.random.key(42))  # the same
weights, particles = bz.generate(n_events=1000)                  # fresh key, not reproducible
```

Passing the same key twice returns the very same events.

### Compilation

The generation is JIT-compiled with `jax.jit`. The number of events is a *static* argument, so a
call with a new `n_events` triggers a recompilation while repeated calls with the same value reuse
the compiled function:

```python
for i in range(10):
    weights, particles = bz.generate(n_events=1000, key=i)
```

Setting the environment variable `PHASESPACE_EAGER=1` (or calling `jax.disable_jit()`) makes
everything run eagerly, which is useful when debugging the internals.

Importing `phasespace` enables JAX's double precision mode
(`jax.config.update("jax_enable_x64", True)`) process-wide, as the phase space computation is not
numerically stable in single precision.

More examples can be found in the `tests` folder and in the
[documentation](https://phasespace.readthedocs.io/en/latest/usage.html).

## Physics validation

Physics validation is performed continuously in the included tests (`tests/test_physics.py`), run
through GitHub Actions. This validation is performed at two levels:

- In simple `n`-body decays, the results of `phasespace` are checked against `TGenPhaseSpace`.
- For sequential decays, the results of `phasespace` are checked against
  [RapidSim](https://github.com/gcowan/RapidSim/), a "fast Monte Carlo generator for simulation of
  heavy-quark hadron decays".
  In the case of resonances, differences are expected because our tests don't include proper
  modelling of their mass shape, as it would require the introduction of further dependencies.
  However, the results of the comparison can be inspected visually.

The results of all physics validation performed by the `test_physics.py` test are written in
`tests/plots`.
