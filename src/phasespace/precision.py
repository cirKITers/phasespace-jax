"""Double precision handling.

The phase space computation asks for ``float64`` explicitly everywhere, but JAX only honours that
while its x64 mode is enabled: with the mode off, every request is truncated to ``float32`` and
warns once per call. Single precision is not an option here, as ``pdk`` suffers catastrophic
cancellation close to threshold, which degrades energy-momentum conservation from ~1e-15 to ~1e-6
relative.

The mode is therefore enabled per call rather than at import time, which keeps the dtype defaults
of the calling program untouched.
"""

import functools

import jax


def with_float64(func):
    """Run ``func`` with JAX's double precision mode enabled.

    Enabling the mode is scoped to the call, so it does not change the dtype defaults of the
    calling program. The returned arrays are ``float64``.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with jax.enable_x64():
            return func(*args, **kwargs)

    return wrapper
