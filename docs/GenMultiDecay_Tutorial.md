<!-- Generated from GenMultiDecay_Tutorial.ipynb. Edit the notebook, not this file. -->

# Tutorial for the *GenMultiDecay* class
`phasespace.fromdecay.GenMultiDecay` generates events from decay chains with multiple decay modes.

Install the optional dependencies with
`pip install phasespace-jax[fromdecay]`.

The `fromdecay` submodule connects `phasespace` with [`DecayLanguage`](https://github.com/scikit-hep/decaylanguage/).
`GenMultiDecay` also provides an interface for particles with several decay modes.

```python
# Import libraries
from pprint import pprint

import jax
import jax.numpy as jnp
from particle import Particle
from decaylanguage import DecFileParser, DecayChainViewer, DecayChain, DecayMode

from phasespace.fromdecay import GenMultiDecay
```

## Introduction to DecayLanguage
DecayLanguage parses and displays `.dec` files, which define particle decay modes and their branching fractions. For details, see the [DecayLanguage documentation](https://github.com/scikit-hep/decaylanguage).

First, parse a `.dec` file:

```python
parser = DecFileParser("../tests/fromdecay/example_decays.dec")
parser.parse()
```

Use `parser.build_decay_chains` to obtain a dictionary describing the selected particle and its descendants:

```python
pi0_chain = parser.build_decay_chains("pi0")
pprint(pi0_chain)
```

`DecayChainViewer` displays the decay chain as a diagram:

```python
DecayChainViewer(pi0_chain)
```

Alternatively, construct a decay chain with `DecayChain` and `DecayMode`. The following example specifies one decay mode for each decaying particle:

```python
dplus_decay = DecayMode(1, "K- pi+ pi+ pi0", model="PHSP")
pi0_decay = DecayMode(1, "gamma gamma")
dplus_single = DecayChain("D+", {"D+": dplus_decay, "pi0": pi0_decay})
DecayChainViewer(dplus_single.to_dict())
```

## Creating a GenMultiDecay object
The parsed chain contains four decay modes for $\pi^0$. Create a `GenMultiDecay` object from its DecayLanguage dictionary to sample among those modes:

```python
pi0_decay = GenMultiDecay.from_dict(pi0_chain)
```

`GenMultiDecay.from_dict` converts each complete decay path into a separate `GenParticle` instance.

The `gen_particles` attribute stores these instances as `(probability, GenParticle)` pairs:

```python
for probability, particle in pi0_decay.gen_particles:
    print(
        f"There is a probability of {probability} "
        f"that pi0 decays into {', '.join(child.name for child in particle.children)}"
    )
```

Call `generate` to sample events across the available decay paths.

The method draws a decay path for each event, generates events for each selected path and returns lists of weights and particle momenta. Paths with zero selected events are omitted.

```python
weights, events = pi0_decay.generate(n_events=10_000)
print("Number of events for each decay mode:", ", ".join(str(len(w)) for w in weights))
```

The event counts should approximately follow the listed probabilities.

## Changing mass settings
DecayLanguage dictionaries do not specify particle masses. Therefore, `fromdecay` looks up masses by particle name using the [particle](https://github.com/scikit-hep/particle) package.
It assigns a fixed mass or a mass function to each particle; the top particle always has a fixed mass.
The optional `tolerance`, `mass_converter` and `particle_model_map` arguments to `GenMultiDecay.from_dict` control this behavior.

### Fixed and variable masses
A decaying particle receives a fixed mass when its width is at most `tolerance`. The following decay illustrates this rule:

```python
dsplus_chain = parser.build_decay_chains("D*+", stable_particles=["D+"])
DecayChainViewer(dsplus_chain)
```

```python
print(
    f"pi0 width = {Particle.from_evtgen_name('pi0').width}\n"
    f"D0 width = {Particle.from_evtgen_name('D0').width}"
)
```

$\pi^0$ has a greater width than $D^0$.
A tolerance between their widths therefore assigns a fixed mass to $D^0$ and a variable mass to $\pi^0$.

```python
dstar_decay = GenMultiDecay.from_dict(dsplus_chain, tolerance=1e-8)
# Loop over D0 and pi+ particles, see graph above
for particle in dstar_decay.gen_particles[0][1].children:
    # If a particle width is less than tolerance or if it does not have any children, its mass will be fixed.
    assert particle.has_fixed_mass

# Loop over D+ and pi0. See above.
for particle in dstar_decay.gen_particles[1][1].children:
    if particle.name == "pi0":
        assert not particle.has_fixed_mass
```

### Configuring mass functions
Variable masses use a relativistic Breit-Wigner distribution by default. To select a different function for one decay mode, add a `zfit` field to that mode in the DecayLanguage dictionary. For example, in the preceding $D^{*+}$ chain:

```python
dsplus_custom_mass_func = dsplus_chain.copy()
dsplus_chain_subset = dsplus_custom_mass_func["D*+"][1]["fs"][1]
print("Before:")
pprint(dsplus_chain_subset)
# Set the mass function of pi0 to a gaussian distribution when it decays into two photons (gamma)
dsplus_chain_subset["pi0"][0]["zfit"] = "gauss"
print("After:")
pprint(dsplus_chain_subset)
```

The added `zfit` field selects a Gaussian mass function for the first $\pi^0$ decay mode. Pass the modified dictionary to `GenMultiDecay.from_dict`:

```python
GenMultiDecay.from_dict(dsplus_custom_mass_func)
```

To assign the same mass function to every $\pi^0$ decay mode, pass `particle_model_map` to `from_dict`:

```python
GenMultiDecay.from_dict(
    dsplus_chain, particle_model_map={"pi0": "gauss"}
)  # pi0 always decays with a gaussian mass distribution.
```

With `DecayChain`, specify the function directly in a `DecayMode`:

```python
dplus_decay = DecayMode(
    1, "K- pi+ pi+ pi0", model="PHSP"
)  # The model parameter will be ignored by GenMultiDecay
pi0_decay = DecayMode(
    1, "gamma gamma", zfit="gauss"
)  # Make pi0 have a gaussian mass distribution
dplus_single = DecayChain("D+", {"D+": dplus_decay, "pi0": pi0_decay})
GenMultiDecay.from_dict(dplus_single.to_dict())
```

#### Custom mass functions
The built-in mass function names are `gauss` (Gaussian), `bw` (Breit-Wigner) and `relbw` (relativistic Breit-Wigner).

If neither `zfit` nor `particle_model_map` selects a function, `relbw` is used. Change `GenMultiDecay.DEFAULT_MASS_FUNC` to select another default, such as `"gauss"`. An unknown function name raises `KeyError`.

To add a mass function, define a factory that accepts a particle's mass and width and returns a function with the required [sampling signature](https://cirkiters.github.io/phasespace-jax/usage/#resonances-with-variable-mass). For example:

```python
def custom_gauss(mass, width):
    # This is the actual mass function that will be returned
    def mass_func(min_mass, max_mass, n_events, key):
        # a normal distribution truncated to the kinematically allowed range
        standard = jax.random.truncated_normal(
            key,
            lower=(min_mass - mass) / width,
            upper=(max_mass - mass) / width,
            shape=(n_events,),
            dtype=jnp.float64,
        )
        return mass + width * standard

    return mass_func
```

Pass the factory through `mass_converter`, keyed by the name used in `zfit`. The key can be any name; here it is `"custom_gauss"`:

```python
dsplus_chain_subset = dsplus_custom_mass_func["D*+"][1]["fs"][1]
print("Before:")
pprint(dsplus_chain_subset)

# Set the mass function of pi0 to the custom gaussian distribution
#  when it decays into an electron-positron pair and a photon (gamma)
dsplus_chain_subset["pi0"][1]["zfit"] = "custom_gauss"
print("After:")
pprint(dsplus_chain_subset)
```

```python
GenMultiDecay.from_dict(dsplus_custom_mass_func, {"custom_gauss": custom_gauss})
```
