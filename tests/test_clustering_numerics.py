"""Numerical equivalence of the vectorized DP probability reductions."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from Cluster.DpEngine import DpEngine
from data.Patient import NDHistogram


@pytest.mark.parametrize("samples,bins", [(1, 51), (4, 101), (20, 201)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reductions_match_rowwise_reference(samples, bins, dtype):
    rng = np.random.default_rng(7)
    # Strided, read-only inputs exercise both the copying and zero-copy paths.
    loghist = rng.uniform(-10000, 1000, (samples, bins * 2)).astype(dtype)[:, ::2]
    loghist[:, ::7] = -np.inf
    prior = np.log(rng.dirichlet(np.ones(bins), size=samples)).astype(dtype)
    original_hist = loghist.copy()
    original_prior = prior.copy()
    loghist.setflags(write=False)
    prior.setflags(write=False)
    engine = SimpleNamespace(logprior=prior)

    marginal = DpEngine.logsum_of_marginals_per_sample(loghist)
    expected_marginal = np.array([logsumexp(row) for row in loghist.astype(np.float32)])
    assert marginal.shape == (samples,)
    assert marginal.dtype == np.float32
    np.testing.assert_allclose(marginal, expected_marginal, rtol=1e-6, atol=1e-6)

    normalized = DpEngine.normalize_loghist_with_prior(engine, loghist)
    expected_normalized = np.array([row - logsumexp(row) for row in loghist + prior])
    assert normalized.shape == (samples, bins)
    assert normalized.dtype == dtype
    np.testing.assert_allclose(normalized, expected_normalized, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.exp(normalized).sum(axis=1), 1, atol=1e-4)
    np.testing.assert_array_equal(loghist, original_hist)
    np.testing.assert_array_equal(prior, original_prior)


def test_nonuniform_prior_normalizes_each_sample_independently():
    weights = np.array([[1., 2., 3.], [4., 2., 1.]])
    prior = np.array([[.5, .25, .25], [.2, .3, .5]])
    normalized = DpEngine.normalize_loghist_with_prior(
        SimpleNamespace(logprior=np.log(prior)), np.log(weights))
    expected = weights * prior
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(np.exp(normalized), expected, rtol=1e-12)


def test_seeded_clustering_matches_rowwise_reductions(monkeypatch):
    grid = np.arange(101)
    peaks = [(95, 90, 99, 95)] * 5 + [(40, 60, 30, 50)] * 5
    densities = [[np.exp(-0.5 * ((grid - peak) / 3) ** 2) for peak in sample_peaks]
                 for sample_peaks in peaks]

    def run():
        histogram = NDHistogram(densities, [str(i) for i in range(len(peaks))])
        return DpEngine(histogram, N_iter=12, Pi_k={"r": 3, "mu": 3}, seed=7).results

    def rowwise_marginal(loghist):
        return np.array([logsumexp(row) for row in np.asarray(loghist, dtype=np.float32)])

    def rowwise_normalize(self, loghist):
        return np.array([row - logsumexp(row) for row in loghist + self.logprior])

    with monkeypatch.context() as reference:
        reference.setattr(DpEngine, "logsum_of_marginals_per_sample", staticmethod(rowwise_marginal))
        reference.setattr(DpEngine, "normalize_loghist_with_prior", rowwise_normalize)
        expected = run()
    actual = run()

    np.testing.assert_array_equal(actual.assign, expected.assign)
    np.testing.assert_array_equal(actual.K, expected.K)
    np.testing.assert_allclose(actual.alpha, expected.alpha, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual.var_loc, expected.var_loc, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual.clust_CCF_dens, expected.clust_CCF_dens, rtol=1e-6, atol=1e-7)
