"""Distribution comparisons that retain bin positions and weight uncertainties."""

import numpy as np
from scipy.stats import chi2


def histogram_pvalue(first, second, range_, weights_first=None, weights_second=None):
    """ROOT's two-weighted-histogram chi-square test, with adjacent sparse bins merged.

    See https://root.cern.ch/doc/master/classTH1.html, 'Two weighted histograms comparison'.
    The variance of a weighted bin is its sum of squared weights. Unit weights also cover
    unweighted references. Bins are merged to at least ten pooled expected effective entries
    in each sample before applying the asymptotic test.
    """
    histograms = []
    variances = []
    for values, weights in ((first, weights_first), (second, weights_second)):
        values = np.asarray(values)
        weights = np.ones_like(values) if weights is None else np.asarray(weights)
        histogram = np.histogram(values, bins=100, range=range_, weights=weights)[0]
        variance = np.histogram(values, bins=100, range=range_, weights=weights**2)[0]
        total = histogram.sum()
        assert total > 0, "No events in the comparison range"
        histograms.append(histogram / total)
        variances.append(variance / total**2)

    min_probability = 10 * max(v.sum() for v in variances)
    groups = []
    current = np.zeros(4)
    for bins in zip(*histograms, *variances):
        current += bins
        if (current[0] + current[1]) / 2 >= min_probability:
            groups.append(current)
            current = np.zeros(4)
    if groups:
        groups[-1] += current
    else:
        groups.append(current)
    if len(groups) == 1:
        return 1.0
    first, second, variance_first, variance_second = np.asarray(groups).T
    statistic = np.sum((first - second) ** 2 / (variance_first + variance_second))
    return chi2.sf(statistic, len(groups) - 1)
