"""This submodule makes it possible for `phasespace` and `DecayLanguage` to work together.

More generally, the `GenMultiDecay` object can also be used as a high-level interface for simulating particles
that can decay in multiple different ways.
"""

from __future__ import annotations

from .genmultidecay import GenMultiDecay

try:
    from particle import Particle  # noqa: F401
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "The fromdecay functionality in phasespace requires particle. "
        "Either install phasespace-jax[fromdecay] or particle."
    ) from error

__all__ = ("GenMultiDecay",)


def __dir__() -> tuple[str, ...]:
    return __all__
