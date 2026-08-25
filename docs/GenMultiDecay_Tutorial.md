<!-- Generated from GenMultiDecay_Tutorial.ipynb. Edit the notebook, not this file. -->

# Tutorial for *GenMultiDecay* class
This tutorial shows how `phasespace.fromdecay.GenMultiDecay` can be used.

In order to use this functionality, you need to install the extra dependencies, for example through
`pip install phasespace-jax[fromdecay]`.

This submodule makes it possible for `phasespace` and [`DecayLanguage`](https://github.com/scikit-hep/decaylanguage/) to work together.
More generally, `GenMultiDecay` can also be used as a high-level interface for simulating particles that can decay in multiple different ways.

```python
# Import libraries
from pprint import pprint

import jax
import jax.numpy as jnp
from particle import Particle
from decaylanguage import DecFileParser, DecayChainViewer, DecayChain, DecayMode

from phasespace.fromdecay import GenMultiDecay
```

## Quick Intro to DecayLanguage
DecayLanguage can be used to parse and view .dec files. These files contain information about how a particle decays and with which probability. For more information about DecayLanguage and .dec files, see the [DecayLanguage](https://github.com/scikit-hep/decaylanguage) documentation.

We will begin by parsing a .dec file using DecayLanguage:

```python
parser = DecFileParser("../tests/fromdecay/example_decays.dec")
parser.parse()
```

From the `parser` variable, one can access a certain decay for a particle using `parser.build_decay_chains`. This will be a `dict` that contains all information about how the mother particle, daughter particles etc. decay.

```python
pi0_chain = parser.build_decay_chains("pi0")
pprint(pi0_chain)
```

This `dict` can also be displayed in a more human-readable way using `DecayChainViewer`:

```python
DecayChainViewer(pi0_chain)
```

You can also create a decay using the `DecayChain` and `DecayMode` classes. However, a DecayChain can only contain one chain, i.e., a particle cannot decay in multiple ways.

```python
dplus_decay = DecayMode(1, "K- pi+ pi+ pi0", model="PHSP")
pi0_decay = DecayMode(1, "gamma gamma")
dplus_single = DecayChain("D+", {"D+": dplus_decay, "pi0": pi0_decay})
DecayChainViewer(dplus_single.to_dict())
```

## Creating a GenMultiDecay object
A regular `phasespace.GenParticle` instance would not be able to simulate this decay, since the $\pi^0$ particle can decay in four different ways. However, a `GenMultiDecay` object can be created directly from a DecayLanguage dict:

```python
pi0_decay = GenMultiDecay.from_dict(pi0_chain)
```

When creating a `GenMultiDecay` object, the DecayLanguage dict is "unpacked" into separate GenParticle instances, where each GenParticle instance corresponds to one way that the particle can decay.

These GenParticle instances and the probabilities of that decay mode can be accessed via `GenMultiDecay.gen_particles`. This is a list of tuples, where the first element in the tuple is the probability and the second element is the GenParticle.

```python
for probability, particle in pi0_decay.gen_particles:
    print(
        f"There is a probability of {probability} "
        f"that pi0 decays into {', '.join(child.name for child in particle.children)}"
    )
```

One can simulate this decay using the `.generate` method, which works the same as the `GenParticle.generate` method.

When calling the `GenMultiDecay.generate` method, it internally calls the generate method on the of the GenParticle instances in `GenMultiDecay.gen_particles`. The outputs are placed in a list, which is returned.

```python
weights, events = pi0_decay.generate(n_events=10_000)
print("Number of events for each decay mode:", ", ".join(str(len(w)) for w in weights))
```

We can confirm that the counts above are close to the expected counts based on the probabilities.

## Changing mass settings
Since DecayLanguage dicts do not contain any information about the mass of a particle, the `fromdecay` submodule uses the [particle](https://github.com/scikit-hep/particle) package to find the mass of a particle based on its name.
The mass can either be a constant value or a function (besides the top particle, which is always a constant).
These settings can be modified by passing in additional parameters to `GenMultiDecay.from_dict`.
There are two optional parameters that can be passed to `GenMultiDecay.from_dict`: `tolerance` and `mass_converter`.

### Constant vs variable mass
If a particle has a width less than `tolerance`, its mass is set to a constant value.
This will be demonsttrated with the decay below:

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
If the tolerance is set to a value between their widths, the $D^0$ particle will have a constant mass while $\pi^0$ will not.

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
By default, the mass function used for variable mass is the relativistic Breit-Wigner distribution. This can however be changed. If you want the mother particle to have a specific mass function for a specific decay, you can add a `zfit` parameter to the DecayLanguage dict. Consider for example the previous $D^{*+}$ example:

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

Notice the added `zfit` field to the first decay mode of the $\pi^0$ particle. This dict can then be passed to `GenMultiDecay.from_dict`, like before.

```python
GenMultiDecay.from_dict(dsplus_custom_mass_func)
```

If you want all $\pi^0$ particles to decay with the same mass function, you do not need to specify the `zfit` parameter for each decay in the `dict`. Instead, one can pass the `particle_model_map` parameter to the constructor:

```python
GenMultiDecay.from_dict(
    dsplus_chain, particle_model_map={"pi0": "gauss"}
)  # pi0 always decays with a gaussian mass distribution.
```

When using `DecayChain`s, the syntax for specifying the mass function becomes cleaner:

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
The built-in supported mass function names are `gauss`, `bw`, and `relbw`, with `gauss` being the gaussian distribution, `bw` being the Breit-Wigner distribution, and `relbw` being the relativistic Breit-Wigner distribution.

If a non-supported value for the `zfit` parameter is not specified, it will automatically use the relativistic Breit-Wigner distribution. This behavior can be changed by changing the value of `GenMultiDecay.DEFAULT_MASS_FUNC` to a different string, e.g., `"gauss"`. If an invalid value for the `zfit` parameter is used, a `KeyError` is raised.

It is also possible to add your own mass functions besides the built-in ones. You should then create a function that takes the mass and width of a particle and returns a mass function which with the [format](https://cirkiters.github.io/phasespace-jax/usage/#resonances-with-variable-mass) that is used for all phasespace mass functions. Below is an example of a custom gaussian distribution, implemented in the same way as the built-in one:

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

This function can then be passed to `GenMultiDecay.from_dict` as a dict, where the key specifies the `zfit` parameter name. In the example below, it is set to `"custom_gauss"`. However, this name can be chosen arbitrarily and does not need to be the same as the function name.

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
