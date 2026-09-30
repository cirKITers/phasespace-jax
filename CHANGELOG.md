# Changelog

Entries through version 1.10.0 document the history of the upstream
[zfit/phasespace](https://github.com/zfit/phasespace) project.

## Develop

### Major Features and Improvements
- Fixed Lorentz boosts for batches containing both stationary and moving parents. Internal boosts
  use momentum and invariant mass directly, preserving conservation accuracy for light or
  massless daughters. Corrected `kinematics.beta` to return speed divided by the speed of light.
- Rotation coefficients are computed once per stage, avoiding repeated transcendental
  evaluations in compiled CPU kernels. Paired CPU and GPU benchmarks show faster generation for
  large batches. Precision and random key splitting are unchanged.
- Relativistic Breit-Wigner sampling uses bounded rejection sampling when the requested interval
  falls outside the CDF table or the table cannot resolve it. Sampling within the table remains
  unchanged.
- Physics validation now checks four-momentum conservation, mass shells, massless energies,
  angular distributions and resonance tails. Reference comparisons preserve histogram bin
  positions and account for weighted statistical uncertainties.
- Generation runs on a GPU without code changes when a CUDA-enabled `jaxlib` is installed
  (`pip install "jax[cuda12]"` for Maxwell to Volta, `"jax[cuda13]"` from Turing on).
  Speedup depends on event count and device. CPU and GPU results agree within a few units in the
  last place (ULPs), although arithmetic is not bit-identical. The PRNG is backend independent.
- `generate` accepts `chunk_size` to generate events in batches and limit peak memory
  use. Each chunk receives a separate split of `key`, so a chunked run draws a different but
  equally reproducible sample.
- Replaced `jnp.sort` with an explicit compare-exchange network for the short axis of
  `n_particles - 2` entries. XLA previously used a general sort for this axis. Generated
  events remain bit-identical.
- Ported the computational backend from TensorFlow to JAX. Compilation with `jax.jit` makes
  `generate` roughly four to five times faster on CPU for one million `B -> 3pi` events.
- The `phasespace.fromdecay` mass functions (`gauss`, `bw`, `relbw`) now sample
  directly with JAX using inverse transform sampling. This removes the zfit and zfit-physics
  dependencies.

### Behavioral changes
- Decays at the exact threshold raise `ValueError` because their normalized weights are
  undefined. Numerical boost corrections can change the last bits of generated momenta.
  Generation remains reproducible for a fixed key; phase-space key splitting and weight formulas
  are unchanged.
- `generate` accepts `key` in place of `seed`. The value can be an integer, a JAX PRNG
  key or `None`. JAX random generation is functional: reusing a key produces identical events,
  whereas `tf.random.Generator` advances its state between calls.
- `n_events` must be a Python integer and is a static argument of the compiled function.
  A new value triggers compilation. `tf.Variable` is no longer accepted.
- Resonance mass functions receive `mass(min_mass, max_mass, n_events, key)` and must be
  compatible with `jax.jit`. Signature inspection that conditionally passed `seed` has been
  removed.
- Kinematically forbidden decays raise `ValueError` instead of
  `tf.errors.InvalidArgumentError`.
- `generate` enables JAX double precision for the duration of the call and returns `float64`
  arrays, regardless of the caller's setting. Single precision is numerically unstable for this
  computation. Importing `phasespace` leaves global JAX settings unchanged. With x64 mode
  disabled, later JAX operations can downcast the arrays to `float32` and issue a warning;
  conversion to NumPy preserves their precision. The helpers in `phasespace.kinematics`
  follow the caller's precision to remain compatible with `jax.jit`.
- `GenMultiDecay.generate` accepts a `key` argument. Previously, decay mode selection
  used the global TensorFlow seed and ignored the supplied seed.
- Removed the `generate_tensor`, `Particle` and `generate_decay` stubs, which only raised.
- `phasespace.numpy` is now `jax.numpy` instead of `tensorflow.experimental.numpy`.

### Bug fixes and small changes
- Resonance masses are now drawn from the key passed to `generate`. Previously, they were
  drawn from the global TensorFlow generator, so seeded decays with resonances were not
  reproducible.
- `PHASESPACE_EAGER=0` now leaves eager mode disabled. Previously, it was interpreted as a
  non-empty string and enabled eager mode.
- The `fromdecay` import error no longer passes an invalid `file` keyword to
  `ModuleNotFoundError`. That keyword previously caused a `TypeError` and obscured the
  intended message.

### Requirement changes
- Requires `jax >= 0.11.0`. `tensorflow` and `tensorflow_probability` are no longer required,
  and the `tf`/`tensorflow` extras were removed.
- Requires Python >= 3.12, the minimum supported by JAX 0.11. Support for 3.10 and 3.11 is dropped.
- The `fromdecay` extra no longer requires `zfit` and `zfit-physics`.


## 1.10.0 (16 Apr 2024)

Add support for Python 3.12, drop support for 3.8

### Major Features and Improvements

- integrating [vector](https://vector.readthedocs.io/en/latest/index.html) support for `generate`: `boost_to` can be a Momentum Lorentz vector and return the boosted particles as a vector using `as_vectors=True`.

### Requirement changes

Upgrade to TensorFlow > 0.16

## 1.9.0 (20 Jul 2023)

Add support for Python 3.11, drop support for 3.7

## 1.8.0 (27 Jan 2023)

### Requirement changes
- upgrade to zfit >= 0.10.0 and zfit-physics >= 0.3.0
- pinning uproot and awkward to ~4 and ~1, respectively

## 1.7.0 (1. Sep 2022)

Upgraded Python and TensorFlow version.

Added `tf` and `tensorflow` extra to requirements. If you intend to use
phasespace with TensorFlow in the future (and not another backend like numpy or JAX),
make sure to always install with `phasespace[tf]`.

### Requirement changes
- upgrade to TensorFlow >= 2.7
- Python from 3.7 to 3.10 is now supported

## 1.6.0 (14 Apr 2022)

### Major Features and Improvements
- Improved GenMultiDecay to have better control on the decay mass of non-stable particles.
- Added a `particle_model_map` argument to the `GenMultiDecay` class. This is a
  dict where the key is a particle name and the value is a mass function name.
  The feature can be seen in the
  [GenMultiDecay Tutorial](https://github.com/zfit/phasespace/blob/master/docs/GenMultiDecay_Tutorial.ipynb).

## 1.5.0 (27 Nov 2021)

### Major Features and Improvements
- add support to generate from a DecayChain using
  [the decaylanguage](https://github.com/scikit-hep/decaylanguage) package from Scikit-HEP.
  This is in the new subpackage "fromdecay" and can be used by installing the extra with
  `pip install phasespace[fromdecay]`.

### Requirement changes
- drop Python 3.6 support

### Thanks
- to Simon Thor for contributing the `fromdecay` subpackage.

## 1.4.2 (5.11.2021)

### Requirement changes
- Losen restriction on TensorFlow, allow version 2.7 (and 2.5, 2.6)

## 1.4.1 (27.08.2021)

### Requirement changes
- Losen restriction on TensorFlow, allow version 2.6 (and 2.5)

## 1.4.0 (11.06.2021)

### Requirement changes
- require TensorFlow 2.5 as 2.4 breaks some functionality

## 1.3.0 (28.05.2021)

### Major Features and Improvements

- Support Python 3.9
- Support TensorFlow 2.5
- improved compilation in tf.functions, use of XLA where applicable
- developer: modernization of setup, CI and more

### Thanks

- Remco de Boer for many commits and cleanups

## 1.2.0 (17.12.20)

### Major Features and Improvements

- Python 3.8 support
- Allow eager execution by setting with `tf.config.run_functions_eagerly(True)`
  or the environment variable "PHASESPACE_EAGER"
- Deterministic random number generation via seed
  or `tf.random.Generator` instance

### Behavioral changes

### Bug fixes and small changes

### Requirement changes

- tighten TensorFlow to 2.3/2.4
- tighten TensorFlow Probability to 0.11/0.12

### Thanks
- Remco de Boer and Stefan Pflüger for discussions on random number genration

## 1.1.0 (27.1.2020)

This release switched to TensorFlow 2.0 eager mode. Please upgrade your TensorFlow installation if possible and change
your code (minimal changes) as described under "Behavioral changes".
In case this is currently impossible to do, please downgrade to < 1.1.0.

### Major Features and Improvements
- full TF2 compatibility

### Behavioral changes
- `generate` now returns an eager Tensor. This is basically a numpy array wrapped by TensorFlow.
   To explicitly convert it to a numpy array, use the `numpy()` method of the eager Tensor.
- `generate_tensor` is now depreceated, `generate` can directly be used instead.

### Bug fixes and small changes

### Requirement changes
- requires now TensorFlow >= 2.0.0


## 1.0.4 (13-10-2019)

### Major Features and Improvements

Release to conda-forge, thanks to Chris Burr
