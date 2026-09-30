# phasespace-jax

`phasespace-jax` generates $n$-body phase space events with the Raubold and Lynch method
(GENBOD, [CERN-68-15](https://cds.cern.ch/record/275743)), using [JAX](https://github.com/jax-ml/jax)
as its computational backend.

This fork replaces the TensorFlow backend of
[zfit/phasespace](https://github.com/zfit/phasespace) with JAX and supports compilation with
`jax.jit`. It retains the generation algorithm, public API and `DecayLanguage` integration.
However, TensorFlow and JAX use different random streams, so the same integer seed does not
produce bit-identical events across the two packages.

## Installation

```console
$ pip install phasespace-jax
```

For the [DecayLanguage](https://github.com/scikit-hep/decaylanguage) integration:

```console
$ pip install "phasespace-jax[fromdecay]"
```

The distribution is named `phasespace-jax`, while its import name remains `phasespace`. It
therefore cannot be installed alongside the upstream `phasespace` distribution.

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

See [Usage](usage.md) for decay chains, resonances, boosts and JIT behavior. The
[DecayChain tutorial](GenMultiDecay_Tutorial.md) shows how to build decays from `.dec` files.

## Differences from the upstream package

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

## Physics validation

Physics validation runs in `tests/test_physics.py`. The tests compare simple $n$-body decays with
`TGenPhaseSpace` and sequential decays with [RapidSim](https://github.com/gcowan/RapidSim/).
Resonance comparisons can differ because these tests do not include the full mass-shape model
used by the reference generator. Comparison plots are written to `tests/plots`.

## Citing

The underlying algorithm comes from the upstream package; please cite the original work:

> A. Puig Navarro and J. Eschle, *phasespace: n-body phase space generation in Python*,
> Journal of Open Source Software **4**(42), 1570 (2019),
> [doi:10.21105/joss.01570](https://doi.org/10.21105/joss.01570).

`phasespace-jax` is distributed under the upstream project's BSD-3-Clause license and retains its
copyright notice. See [Authors](authors.md). This fork is not affiliated with or endorsed by the
zfit project.
