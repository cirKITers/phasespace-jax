#!/usr/bin/env python3
# =============================================================================
# @file   test_physics.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   27.02.2019
# =============================================================================
"""Test physics output."""

import platform
import subprocess

import numpy as np
import pytest
import vector

if platform.system() == "Darwin":
    import matplotlib

    matplotlib.use("TkAgg")

import os
import sys

import matplotlib.pyplot as plt
import uproot

from phasespace import phasespace

sys.path.append(os.path.dirname(__file__))

from .helpers import decays, rapidsim
from .helpers.plotting import make_norm_histo
from .helpers.statistics import histogram_pvalue

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOT_DIR = os.path.join(BASE_PATH, "tests", "plots")


def create_ref_histos(n_pions):
    """Load reference histogram data."""
    ref_dir = os.path.join(BASE_PATH, "data")
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    ref_file = os.path.join(ref_dir, f"bto{n_pions}pi.root")
    if not os.path.exists(ref_file):
        script = os.path.join(
            BASE_PATH,
            "scripts",
            "prepare_test_samples.cxx+({})".format(
                ",".join('"{}"'.format(os.path.join(BASE_PATH, "data", f"bto{i + 1}pi.root")) for i in range(1, 4))
            ),
        )
        subprocess.call(f"root -qb '{script}'", shell=True)
    events = uproot.open(ref_file)["events"]
    pion_names = [f"pion_{pion + 1}" for pion in range(n_pions)]
    pions = {pion_name: events[pion_name] for pion_name in pion_names}
    weights = events["weight"].array(library="np")
    normalized_histograms = []
    samples = []
    for pion in pions.values():
        pion_array = pion.array()
        energy = pion_array.fE
        momentum = pion_array.fP
        for coord, array in enumerate([momentum.fX, momentum.fY, momentum.fZ, energy]):
            numpy_array = np.array(array)
            samples.append(numpy_array)
            histogram = make_norm_histo(
                numpy_array,
                range_=(-3000 if coord % 4 != 3 else 0, 3000),
                weights=weights,
            )
            normalized_histograms.append(histogram)

    return normalized_histograms, make_norm_histo(weights, range_=(0, 1 + 1e-8)), samples, weights


def run_test(n_particles, test_prefix):
    first_run_n_events = 100
    main_run_n_events = 100000

    decay = phasespace.nbody_decay(decays.B0_MASS, [decays.PION_MASS] * n_particles)
    weights1, _ = decay.generate(first_run_n_events)  # triggers a first compilation
    assert len(weights1) == first_run_n_events

    # a different n_events recompiles and must give consistent results
    weights, particles = decay.generate(main_run_n_events)
    parts = np.concatenate([particles[f"p_{part_num}"] for part_num in range(n_particles)], axis=1)
    histos = [
        make_norm_histo(
            parts[:, coord],
            range_=(-3000 if coord % 4 != 3 else 0, 3000),
            weights=weights,
        )
        for coord in range(parts.shape[1])
    ]
    weight_histos = make_norm_histo(weights, range_=(0, 1 + 1e-8))
    ref_histos, ref_weights, ref_samples, ref_event_weights = create_ref_histos(n_particles)
    p_values = np.array(
        [
            histogram_pvalue(
                parts[:, coord],
                ref_samples[coord],
                range_=(-3000 if coord % 4 != 3 else 0, 3000),
                weights_first=weights,
                weights_second=ref_event_weights,
            )
            for coord, _ in enumerate(histos)
        ]
        + [histogram_pvalue(weights, ref_event_weights, range_=(0, 1 + 1e-8))]
    )
    # Let's plot
    x = np.linspace(-3000, 3000, 100)
    e = np.linspace(0, 3000, 100)
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    for coord, _ in enumerate(histos):
        plt.hist(
            x if coord % 4 != 3 else e,
            weights=histos[coord],
            alpha=0.5,
            label="phasespace",
            bins=100,
        )
        plt.hist(
            x if coord % 4 != 3 else e,
            weights=ref_histos[coord],
            alpha=0.5,
            label="TGenPhasespace",
            bins=100,
        )
        plt.legend(loc="upper right")
        plt.savefig(
            os.path.join(
                PLOT_DIR,
                "{}_pion_{}_{}.png".format(test_prefix, int(coord / 4) + 1, ["px", "py", "pz", "e"][coord % 4]),
            )
        )
        plt.clf()
    plt.hist(
        np.linspace(0, 1, 100),
        weights=weight_histos,
        alpha=0.5,
        label="phasespace",
        bins=100,
    )
    plt.hist(
        np.linspace(0, 1, 100),
        weights=ref_weights,
        alpha=0.5,
        label="phasespace",
        bins=100,
    )
    plt.savefig(os.path.join(PLOT_DIR, f"{test_prefix}_weights.png"))
    plt.clf()
    # Bonferroni correction: 5% significance for the collection of coordinate/weight checks.
    assert np.all(p_values > 0.05 / len(p_values)), p_values


@pytest.mark.flaky(3)  # Stats are limited
def test_two_body():
    """Test B->pipi decay."""
    run_test(2, "two_body")


@pytest.mark.flaky(3)  # Stats are limited
def test_three_body():
    """Test B -> pi pi pi decay."""
    run_test(3, "three_body")


@pytest.mark.flaky(3)  # Stats are limited
def test_four_body():
    """Test B -> pi pi pi pi decay."""
    run_test(4, "four_body")


def run_kstargamma(input_file, kstar_width, b_at_rest, suffix, use_vector):
    """Run B0->K*gamma test."""
    n_events = 1000000
    if b_at_rest:
        booster = None
        rapidsim_getter = rapidsim.get_tree_in_b_rest_frame
    else:
        booster = rapidsim.generate_fonll(decays.B0_MASS, 7, "b", n_events)
        booster = booster.transpose()
        if use_vector:
            booster = vector.array(
                {
                    "px": booster[:, 0],
                    "py": booster[:, 1],
                    "pz": booster[:, 2],
                    "e": booster[:, 3],
                }
            )
        rapidsim_getter = rapidsim.get_tree
    decay = decays.b0_to_kstar_gamma(kstar_width=kstar_width)
    norm_weights, particles = decay.generate(n_events=n_events, boost_to=booster)
    rapidsim_parts = rapidsim_getter(
        os.path.join(BASE_PATH, "data", "large", input_file),
        "B0_0",
        ("Kst0_0", "gamma_0", "Kp_0", "pim_0"),
    )
    name_matching = {"Kst0_0": "K*0", "gamma_0": "gamma", "Kp_0": "K+", "pim_0": "pi-"}
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    x = np.linspace(-3000, 3000, 100)
    e = np.linspace(0, 3000, 100)
    p_values = {}
    for ref_name, ref_part in rapidsim_parts.items():
        tf_part = name_matching[ref_name]
        ref_part = ref_part.transpose()  # for consistency
        for coord, coord_name in enumerate(("px", "py", "pz", "e")):
            range_ = (-3000 if coord % 4 != 3 else 0, 3000)
            ref_histo = make_norm_histo(ref_part[:, coord], range_=range_)
            tf_histo = make_norm_histo(particles[tf_part][:, coord], range_=range_, weights=norm_weights)
            plt.hist(
                x if coord % 4 != 3 else e,
                weights=tf_histo,
                alpha=0.5,
                label="phasespace",
                bins=100,
            )
            plt.hist(
                x if coord % 4 != 3 else e,
                weights=ref_histo,
                alpha=0.5,
                label="RapidSim",
                bins=100,
            )
            plt.legend(loc="upper right")
            plt.savefig(
                os.path.join(
                    PLOT_DIR,
                    "B0_Kstar_gamma_Kstar{}_{}_{}.png".format(suffix, tf_part.replace("*", "star"), coord_name),
                )
            )
            plt.clf()
            p_values[(tf_part, coord_name)] = histogram_pvalue(
                particles[tf_part][:, coord], ref_part[:, coord], range_=range_, weights_first=norm_weights
            )
    plt.hist(
        np.linspace(0, 1, 100),
        weights=make_norm_histo(norm_weights, range_=(0, 1)),
        bins=100,
    )
    plt.savefig(os.path.join(PLOT_DIR, f"B0_Kstar_gamma_Kstar{suffix}_weights.png"))
    plt.clf()
    return np.array(list(p_values.values()))


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.flaky(3)  # Stats are limited
def test_kstargamma_kstarnonresonant_at_rest(vector):
    """Test B0 -> K* gamma physics with fixed mass for K*."""
    p_values = run_kstargamma(
        "B2KstGamma_RapidSim_7TeV_KstarNonResonant_Tree.root",
        0,
        True,
        "NonResonant",
        use_vector=vector,
    )
    assert np.all(p_values > 0.05 / len(p_values)), p_values


@pytest.mark.parametrize("vector", [False, True], ids=["no_vector", "vector"])
@pytest.mark.flaky(3)  # Stats are limited
def test_kstargamma_kstarnonresonant_lhc(vector):
    """Test B0 -> K* gamma physics with fixed mass for K* with LHC kinematics."""
    p_values = run_kstargamma(
        "B2KstGamma_RapidSim_7TeV_KstarNonResonant_Tree.root",
        0,
        False,
        "NonResonant_LHC",
        use_vector=vector,
    )
    assert np.all(p_values > 0.05 / len(p_values)), p_values


def test_kstargamma_resonant_at_rest():
    """Test B0 -> K* gamma physics with Gaussian mass for K*.

    Since we don't have BW and we model the resonances with Gaussians, we can't really perform the Kolmogorov
    test wrt to RapidSim, so plots are generated and can be inspected by the user. However, small differences
    are expected in the tails of the energy distributions of the kaon and the pion.
    """
    run_kstargamma(
        "B2KstGamma_RapidSim_7TeV_Tree.root",
        decays.KSTARZ_WIDTH,
        True,
        "Gaussian",
        use_vector=False,
    )


def run_k1_gamma(input_file, k1_width, kstar_width, b_at_rest, suffix):
    """Run B+ -> K1gamma test."""
    n_events = 1000000
    if b_at_rest:
        booster = None
        rapidsim_getter = rapidsim.get_tree_in_b_rest_frame
    else:
        booster = rapidsim.generate_fonll(decays.B0_MASS, 7, "b", n_events)
        booster = booster.transpose()
        rapidsim_getter = rapidsim.get_tree
    gamma = decays.bp_to_k1_kstar_pi_gamma(k1_width=k1_width, kstar_width=kstar_width)
    norm_weights, particles = gamma.generate(n_events=n_events, boost_to=booster)
    rapidsim_parts = rapidsim_getter(
        os.path.join(BASE_PATH, "data", "large", input_file),
        "Bp_0",
        ("K1_1270_p_0", "Kst0_0", "gamma_0", "Kp_0", "pim_0", "pip_0"),
    )
    name_matching = {
        "K1_1270_p_0": "K1+",
        "Kst0_0": "K*0",
        "gamma_0": "gamma",
        "Kp_0": "K+",
        "pim_0": "pi-",
        "pip_0": "pi+",
    }
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    x = np.linspace(-3000, 3000, 100)
    e = np.linspace(0, 3000, 100)
    p_values = {}
    for ref_name, ref_part in rapidsim_parts.items():
        tf_part = name_matching[ref_name]
        ref_part = ref_part.transpose()  # to be consistent with internal shape (nevents, nobs)
        for coord, coord_name in enumerate(("px", "py", "pz", "e")):
            range_ = (-3000 if coord % 4 != 3 else 0, 3000)
            ref_histo = make_norm_histo(ref_part[:, coord], range_=range_)
            tf_histo = make_norm_histo(particles[tf_part][:, coord], range_=range_, weights=norm_weights)
            plt.hist(
                x if coord % 4 != 3 else e,
                weights=tf_histo,
                alpha=0.5,
                label="phasespace",
                bins=100,
            )
            plt.hist(
                x if coord % 4 != 3 else e,
                weights=ref_histo,
                alpha=0.5,
                label="RapidSim",
                bins=100,
            )
            plt.legend(loc="upper right")
            plt.savefig(
                os.path.join(
                    PLOT_DIR,
                    "Bp_K1_gamma_K1Kstar{}_{}_{}.png".format(suffix, tf_part.replace("*", "star"), coord_name),
                )
            )
            plt.clf()
            p_values[(tf_part, coord_name)] = histogram_pvalue(
                particles[tf_part][:, coord], ref_part[:, coord], range_=range_, weights_first=norm_weights
            )
    plt.hist(
        np.linspace(0, 1, 100),
        weights=make_norm_histo(norm_weights, range_=(0, 1)),
        bins=100,
    )
    plt.savefig(os.path.join(PLOT_DIR, f"Bp_K1_gamma_K1Kstar{suffix}_weights.png"))
    plt.clf()
    return np.array(list(p_values.values()))


@pytest.mark.flaky(3)  # Stats are limited
def test_k1gamma_kstarnonresonant_at_rest():
    """Test B0 -> K1 (->K*pi) gamma physics with fixed-mass resonances."""
    p_values = run_k1_gamma(
        "B2K1Gamma_RapidSim_7TeV_K1KstarNonResonant_Tree.root",
        0,
        0,
        True,
        "NonResonant",
    )
    assert np.all(p_values > 0.05 / len(p_values)), p_values


@pytest.mark.flaky(3)  # Stats are limited
def test_k1gamma_kstarnonresonant_lhc():
    """Test B0 -> K1 (->K*pi) gamma physics with fixed-mass resonances with LHC kinematics."""
    p_values = run_k1_gamma(
        "B2K1Gamma_RapidSim_7TeV_K1KstarNonResonant_Tree.root",
        0,
        0,
        False,
        "NonResonant_LHC",
    )
    assert np.all(p_values > 0.05 / len(p_values)), p_values


def test_k1gamma_resonant_at_rest():
    """Test B0 -> K1 (->K*pi) gamma physics.

    Since we don't have BW and we model the resonances with Gaussians, we can't really perform the Kolmogorov
    test wrt to RapidSim, so plots are generated and can be inspected by the user.
    """
    run_k1_gamma(
        "B2K1Gamma_RapidSim_7TeV_Tree.root",
        decays.K1_WIDTH,
        decays.KSTARZ_WIDTH,
        True,
        "Gaussian",
    )


if __name__ == "__main__":
    test_two_body()
    test_three_body()
    test_four_body()
    test_kstargamma_kstarnonresonant_at_rest()
    test_kstargamma_kstarnonresonant_lhc()
    test_kstargamma_resonant_at_rest()
    test_k1gamma_kstarnonresonant_at_rest()
    test_k1gamma_kstarnonresonant_lhc()
    test_k1gamma_resonant_at_rest()

# EOF
