"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016-2017
"""

import numpy as np
import numpy.testing as npt

from eskmeans.kmeans import KMeans


def test_mean_numerators():

    np.random.seed(1)

    # Generate data
    D = 3           # dimensions
    N = 10          # number of points to generate
    K_true = 4      # the true number of components
    mu_scale = 4.0
    covar_scale = 0.7
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Model
    K = 3
    assignments = np.random.randint(0, K, N)
    kmeans = KMeans(X, K, assignments)

    # Test `mean_numerators`
    n = 0
    for k in range(kmeans.K):
        component_mean = kmeans.mean_numerators[k]/kmeans.counts[k]
        X_k = X[np.where(kmeans.assignments == k)]
        n += X_k.shape[0]

        npt.assert_almost_equal(np.mean(X_k, axis=0), component_mean)
    assert n == N


def test_neg_sqrd_norm():

    np.random.seed(1)

    # Generate data
    D = 4           # dimensions
    N = 11          # number of points to generate
    K_true = 4      # the true number of components
    mu_scale = 4.0
    covar_scale = 0.7
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Model
    K = 5
    assignments = np.random.randint(0, K, N)

    # Make sure we have consequetive values
    for k in range(assignments.max()):
        while len(np.nonzero(assignments == k)[0]) == 0:
            assignments[np.where(assignments > k)] -= 1
        if assignments.max() == k:
            break

    kmeans = KMeans(X, K, assignments)

    # Test `neg_sqrd_norm`
    for i in range(N):
        x_i = X[i, :]
        expected_sqrd_norms = []
        for k in range(kmeans.K):
            component_mean = kmeans.mean_numerators[k]/kmeans.counts[k]
            expected_sqrd_norms.append(np.linalg.norm(x_i - component_mean)**2)
        npt.assert_almost_equal(kmeans.neg_sqrd_norm(i)[:kmeans.K], -np.array(expected_sqrd_norms))


def test_expected_sum_neg_sqrd_norm():

    # Generate data
    D = 5           # dimensions
    N = 20          # number of points to generate
    K_true = 5      # the true number of components
    mu_scale = 5.0
    covar_scale = 0.6
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Model
    K = 6
    assignments = np.random.randint(0, K, N)

    # Make sure we have consequetive values
    for k in range(assignments.max()):
        while len(np.nonzero(assignments == k)[0]) == 0:
            assignments[np.where(assignments > k)] -= 1
        if assignments.max() == k:
            break

    kmeans = KMeans(X, K, assignments)

    # Test `sum_neg_sqrd_norm`
    expected_sum_neg_sqrd_norm = 0.
    for i in range(N):
        x_i = X[i, :]
        expected_sqrd_norms = []
        for k in range(kmeans.K):
            component_mean = kmeans.mean_numerators[k]/kmeans.counts[k]
            expected_sqrd_norms.append(np.linalg.norm(x_i - component_mean)**2)
        expected_sum_neg_sqrd_norm += -np.array(expected_sqrd_norms)[kmeans.assignments[i]]
    npt.assert_almost_equal(kmeans.sum_neg_sqrd_norm(), expected_sum_neg_sqrd_norm)
