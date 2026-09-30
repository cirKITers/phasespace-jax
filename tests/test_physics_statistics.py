import numpy as np

from .helpers.statistics import histogram_pvalue


def test_histogram_comparison_detects_permuted_bins():
    first = np.repeat([0.0, 1.0, 2.0], [1000, 2000, 4000])
    second = np.repeat([0.0, 1.0, 2.0], [4000, 2000, 1000])
    assert histogram_pvalue(first, second, (-0.5, 2.5)) < 1e-10


def test_weighted_comparison_matches_known_distribution():
    rng = np.random.default_rng(7)
    uniform = rng.uniform(-1, 1, 50000)
    reference = 2 * np.sqrt(rng.uniform(size=50000)) - 1
    probability = histogram_pvalue(uniform, reference, (-1, 1), weights_first=uniform + 1)
    assert probability > 0.001
    scaled = histogram_pvalue(uniform, reference, (-1, 1), weights_first=3 * (uniform + 1))
    np.testing.assert_allclose(probability, scaled, rtol=1e-10)
