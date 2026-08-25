# phasespace-jax

`phasespace-jax` generates $n$-body phase space events with the Raubold and Lynch method
(GENBOD, [CERN-68-15](https://cds.cern.ch/record/275743)), using [JAX](https://github.com/jax-ml/jax)
as its computational backend.

It is a fork of [zfit/phasespace](https://github.com/zfit/phasespace) that replaces the TensorFlow
backend with JAX and makes the generation jit-compatible. The algorithm is unchanged and produces
bit-identical events; the public API and the `DecayLanguage` integration are kept, so the package
works almost as a drop-in replacement.

## Installation

```console
$ pip install phasespace-jax
```

For the [DecayLanguage](https://github.com/scikit-hep/decaylanguage) integration:

```console
$ pip install "phasespace-jax[fromdecay]"
```

The distribution is named `phasespace-jax` but the import name stays `phasespace`. It therefore
cannot be installed alongside the upstream `phasespace` distribution.

## A first decay

```python
import phasespace

B0_MASS = 5279.65
PION_MASS = 139.57018
KAON_MASS = 493.677

weights, particles = phasespace.nbody_decay(
    B0_MASS, [PION_MASS, KAON_MASS]
).generate(n_events=1000, key=42)
```

Continue with [Usage](usage.md) for decay chains, resonances, boosting and JIT behaviour, or with
the [DecayChain tutorial](GenMultiDecay_Tutorial.md) to build decays from `.dec` files.

## Differences to the upstream package

| Aspect | [zfit/phasespace](https://github.com/zfit/phasespace) | this fork |
|---|---|---|
| Backend | TensorFlow | JAX |
| Compilation | `tf.function` | `jax.jit`, with `n_events` as a static argument |
| Random numbers | `seed=`, stateful `tf.random.Generator` | `key=`, functional JAX PRNG key |
| `n_events` | `int`, `tf.Tensor` or `tf.Variable` | Python `int`, a new value recompiles |
| Mass functions | `f(min_mass, max_mass, n_events[, seed])`, TFP or zfit PDFs | `f(min_mass, max_mass, n_events, key)`, has to be jit-compatible |
| Resonance shapes in `fromdecay` | zfit / zfit-physics PDFs | sampled directly in JAX, no zfit dependency |
| Forbidden decays | `tf.errors.InvalidArgumentError` | `ValueError` |
| `phasespace.numpy` | `tensorflow.experimental.numpy` | `jax.numpy` |
| Distribution name | `phasespace` | `phasespace-jax`, imported as `phasespace` |
| Speed | reference | ≈4-5x faster for 1M `B -> 3pi` events on CPU |

## Physics validation

Physics validation runs continuously in `tests/test_physics.py`. Simple $n$-body decays are checked
against `TGenPhaseSpace`, sequential decays against
[RapidSim](https://github.com/gcowan/RapidSim/). For resonances, differences are expected because
the tests do not model the mass shape properly; the comparison plots are written to `tests/plots`.

## Citing

The physics is unchanged from the upstream package, so please cite the original work:

> A. Puig Navarro and J. Eschle, *phasespace: n-body phase space generation in Python*,
> Journal of Open Source Software **4**(42), 1570 (2019),
> [doi:10.21105/joss.01570](https://doi.org/10.21105/joss.01570).

`phasespace-jax` is distributed under the same BSD-3-Clause license as the upstream project, whose
copyright notice is retained in full. See [Authors](authors.md). This fork is not affiliated with
or endorsed by the zfit project.
