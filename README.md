# phasespace-jax

[![tests](https://github.com/cirKITers/phasespace-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/cirKITers/phasespace-jax/actions/workflows/ci.yml)
[![docs](https://github.com/cirKITers/phasespace-jax/actions/workflows/docs.yml/badge.svg)](https://cirkiters.github.io/phasespace-jax/)
[![PyPI](https://img.shields.io/pypi/v/phasespace-jax.svg)](https://pypi.org/project/phasespace-jax/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

> This repository is a fork of [zfit/phasespace](https://github.com/zfit/phasespace).
> For background information and citations in scientific publications, consult the original project.
> See also [License and attribution](#license-and-attribution) and [Citing](#citing).

## Differences from the upstream package

This fork replaces [TensorFlow](https://github.com/tensorflow/tensorflow) with
[JAX](https://github.com/jax-ml/jax) and supports compilation with `jax.jit`. It retains the
GENBOD (Raubold–Lynch) algorithm described in CERN-68-15 (see
[Physics validation](#physics-validation)). However, TensorFlow and JAX use different random
streams, so the same integer seed does not produce bit-identical events across the two packages.
The API, return values and `DecayLanguage` integration remain largely compatible with the
upstream package. See the [documentation](https://cirkiters.github.io/phasespace-jax/) for details.

| Aspect | [zfit/phasespace](https://github.com/zfit/phasespace) | this fork |
|---|---|---|
| Backend | TensorFlow | JAX |
| Compilation | `tf.function` | `jax.jit`, with `n_events` as a static argument |
| Random numbers | `seed=`, stateful `tf.random.Generator` | `key=`, functional JAX PRNG key |
| `n_events` | `int`, `tf.Tensor` or `tf.Variable` | Python `int`; a new value triggers compilation |
| Mass functions | `f(min_mass, max_mass, n_events[, seed])`, TFP or zfit PDFs | `f(min_mass, max_mass, n_events, key)`; must be compatible with `jax.jit` |
| Resonance shapes in `fromdecay` | zfit / zfit-physics PDFs | sampled directly in JAX, no zfit dependency |
| Forbidden decays | `tf.errors.InvalidArgumentError` | `ValueError` |
| `phasespace.numpy` | `tensorflow.experimental.numpy` | `jax.numpy` |
| Distribution name | `phasespace` | `phasespace-jax`, imported as `phasespace` |
| Speed | reference | ≈4-5x faster for 1M `B -> 3pi` events on CPU |

Event generation requires double precision; see [JAX treats and traps](#jax-treats-and-traps).

## Installing

To install with pip:

```console
$ pip install phasespace-jax
```

To install the dependencies for [DecayLanguage](https://github.com/scikit-hep/decaylanguage), use:

```console
$ pip install "phasespace-jax[fromdecay]"
```

For GPU use, install a JAX build compatible with your CUDA version, for example:

```console
$ pip install "jax[cuda13]"   # SM 7.5 and newer, Turing onwards (driver >= 580)
```

## How to use

The [DecayChain tutorial](https://cirkiters.github.io/phasespace-jax/GenMultiDecay_Tutorial/)
shows how to generate events from a `DecayChain` using
[DecayLanguage](https://github.com/scikit-hep/decaylanguage).

For a simple $n$-body decay, `nbody_decay` constructs the decay chain from the parent mass and a
list of daughter masses. Particle names are optional. Then, `generate` produces the event sample.
For example, to generate $B^0\to K\pi$:

```python
import phasespace

B0_MASS = 5279.65
PION_MASS = 139.57018
KAON_MASS = 493.677

weights, particles = phasespace.nbody_decay(
    B0_MASS, [PION_MASS, KAON_MASS]
).generate(n_events=1000)
```

Here, `weights` is a `jax.Array` with 1000 entries. The `particles` dictionary contains one
`(1000, 4)` array per particle; each row is a four-momentum. Use `np.asarray(...)` to convert JAX
arrays to NumPy arrays. Events are generated in the parent rest frame. To generate them with a
specified parent momentum, pass it through `boost_to`.

For sequential decays, use `GenParticle` and its `set_children` method. For example, the following
chain describes $B^{0}\to K^{*}\gamma$ followed by $K^*\to K\pi$:

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

Because `set_children` returns the parent particle, the calls can be chained. Here, `particles`
is a dictionary keyed by particle name:

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

Passing the same key twice returns the same events.

### JAX Treats and Traps

Event generation is compiled with `jax.jit`. The number of events is a *static* argument: a new
`n_events` value triggers compilation, whereas repeated calls with the same value reuse the
compiled function:

```python
for i in range(10):
    weights, particles = bz.generate(n_events=1000, key=i)
```

Set `PHASESPACE_EAGER=1` or use `jax.disable_jit()` to run eagerly when debugging.

**Phase space generation requires double precision for numerical stability.**
`generate` enables JAX's double precision mode for the duration of the call and returns `float64`
arrays. Importing `phasespace` does not change the global JAX setting. To retain 64-bit precision
in subsequent JAX operations, enable it for your program:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

With x64 mode disabled, subsequent JAX operations may downcast the arrays to `float32` and issue
a warning. Conversion with `np.asarray(...)` preserves their precision.

### Running on a GPU

Assuming proper [installation](#installing), you can run directly on a GPU via:

```python
import jax

print(jax.devices())  # [CudaDevice(id=0)] once a CUDA-enabled jaxlib is installed

with jax.default_device(jax.devices("gpu")[0]):
    weights, particles = bz.generate(n_events=10_000, key=42)
```


Generation uses `float64` throughout (see [JAX treats and traps](#jax-treats-and-traps)). GPU
double precision performance varies by device and event count. Memory use can also be limiting;
`chunk_size` divides generation into smaller batches:

```python
weights, particles = bz.generate(n_events=10_000_000, key=42, chunk_size=1_000_000)
```

See the [GPU documentation](https://cirkiters.github.io/phasespace-jax/usage/#running-on-a-gpu)
for details.

More examples can be found in the `tests` folder and in the
[documentation](https://cirkiters.github.io/phasespace-jax/usage/).

## Physics validation

The included physics tests (`tests/test_physics.py`) run through GitHub Actions. They compare:

- Simple $n$-body decays with `TGenPhaseSpace`.
- Sequential decays with
  [RapidSim](https://github.com/gcowan/RapidSim/), a "fast Monte Carlo generator for simulation of
  heavy-quark hadron decays".
  Resonance comparisons can differ because these tests do not include the full mass-shape model
  used by the reference generator. The comparison plots allow visual inspection.

The tests write their comparison plots to `tests/plots`.

## Citing

The underlying algorithm comes from the upstream package; please cite the original work:

> A. Puig Navarro and J. Eschle, *phasespace: n-body phase space generation in Python*,
> Journal of Open Source Software **4**(42), 1570 (2019), [doi:10.21105/joss.01570](https://doi.org/10.21105/joss.01570).

The underlying algorithm is described in F. James, *Monte Carlo Phase Space*, CERN-68-15 (1968).
To reference this fork specifically, see [CITATION.cff](CITATION.cff).

## License and attribution

`phasespace-jax` is a derivative work of [zfit/phasespace](https://github.com/zfit/phasespace),
copyright (c) 2019 zfit, and is distributed under the same [BSD-3-Clause license](LICENSE).
The original copyright notice is retained in full. See [AUTHORS.md](AUTHORS.md) for the original authors.

This fork is not affiliated with or endorsed by the zfit project.
