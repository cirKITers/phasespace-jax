#!/usr/bin/env python3
# =============================================================================
# @file   bench_phasespace.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   27.02.2019
# =============================================================================
"""Benchmark phasespace."""

import os
import sys
from timeit import default_timer

import jax

from phasespace import phasespace

sys.path.append(os.path.dirname(__file__))


def memory_usage():
    """Get memory usage of current process in MiB.

    Tries to use :mod:`psutil`, if possible, otherwise fallback to calling
    ``ps`` directly.

    Return:
        float: Memory usage of the current process.
    """
    pid = os.getpid()
    try:
        import psutil

        process = psutil.Process(pid)
        mem = process.memory_info()[0] / float(2**20)
    except ImportError:
        import subprocess

        out = subprocess.Popen(["ps", "v", "-p", str(pid)], stdout=subprocess.PIPE).communicate()[0].split(b"\n")
        vsz_index = out[0].split().index(b"RSS")
        mem = float(out[1].split()[vsz_index]) / 1024
    return mem


# pylint: disable=too-few-public-methods
class Timer:
    """Time the code placed inside its context.

    Taken from http://coreygoldberg.blogspot.ch/2012/06/python-timer-class-context-manager-for.html

    Attributes:
        verbose (bool): Print the elapsed time at context exit?
        start (float): Start time in seconds since Epoch Time. Value set
            to 0 if not run.
        elapsed (float): Elapsed seconds in the timer. Value set to
            0 if not run.

    Arguments:
        verbose (bool, optional): Print the elapsed time at
            context exit? Defaults to False.
    """

    def __init__(self, verbose=False, n=1):
        """Initialize the timer."""
        self.verbose = verbose
        self.n = n
        self._timer = default_timer
        self.start = 0
        self.elapsed = 0

    def __enter__(self):
        self.start = self._timer()
        return self

    def __exit__(self, *args):
        self.elapsed = self._timer() - self.start
        if self.verbose:
            print(f"Elapsed time: {self.elapsed * 1000.0 / self.n} ms")


# EOF


B_MASS = 5279.0
PION_MASS = 139.6

N_EVENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
CHUNK_SIZE = int(N_EVENTS)

n_runs = 10


def test_three_body():
    """Test B -> pi pi pi decay."""
    with Timer(verbose=True):
        print("Initial run (includes the jit compilation, slower than consequent runs)")
        do_run(0)  # to get rid of initial overhead
    print("starting benchmark")
    with Timer(verbose=True, n=n_runs):
        for run in range(n_runs):
            samples = do_run(run + 1)

    print(f"nevents produced {samples[0][0].shape}")
    print("Shape of one particle momentum", samples[0][1]["p_0"].shape)


decay = phasespace.nbody_decay(
    B_MASS,
    [PION_MASS, PION_MASS, PION_MASS],
)


def do_run(run):
    samples = [decay.generate(N_EVENTS, key=run) for _ in range(0, N_EVENTS, CHUNK_SIZE)]
    return jax.block_until_ready(samples)


if __name__ == "__main__":
    # the backend is picked by JAX itself; JAX_PLATFORMS=cpu/cuda selects it for a run
    print(f"{N_EVENTS} events on {jax.devices()}")
    test_three_body()

# EOF
