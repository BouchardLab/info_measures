import numpy as np
from numpy import testing as npt
from scipy.stats import (multivariate_normal, ttest_1samp)

from info_measures.numpy import (kraskov_stoegbauer_grassberger as ksg,
                                 kolchinsky_tracey as kt)


def mvn_entropy(mean, covariance):
    return np.asscalar(multivariate_normal(mean, covariance).entropy())


def test_mvn_kt_entropy():
    """Test Kolchinsky-Tracey entropy estimators on multivariate normal data."""
    n_repeat = 10
    eu = np.zeros(n_repeat)
    el = np.zeros(n_repeat)
    for dim, n_samples in zip([1, 2, 3], [1000, 1000, 10000]):
        mu = np.zeros(dim)
        svar = np.diag(np.sqrt(np.power(10., np.arange(dim) - (dim-1.)/2.)))
        e = mvn_entropy(mu, svar)
        npt.assert_array_less(0., e)
        for ii in range(n_repeat):
            x = np.random.multivariate_normal(mu, svar, size=n_samples)
            kt_estimator = kt.Entropy(x)
            eu[ii] = kt_estimator.entropy(lower=False)
            el[ii] = kt_estimator.entropy(lower=True)
        npt.assert_array_less(e, eu, 'Entropy upper bound was not larger ' +
                                     'than ground truth. Dim: {}'.format(dim))
        npt.assert_array_less(0., el, 'Entropy lower bound was not larger ' +
                                      'than zero. Dim: {}'.format(dim))
        npt.assert_array_less(el, e, 'Entropy lower bound was not smaller ' +
                                     'than ground truth. Dim: {}'.format(dim))


def test_mvn_ksg_entropy():
    """Test Kraskov=Stoegbauer-Grassberger entropy estimator on multivariate normal data."""
    n_repeat = 10
    ek = np.zeros(n_repeat)
    for dim, n_samples in zip([1, 2, 3], [1000, 1000, 10000]):
        mu = np.zeros(dim)
        svar = np.diag(np.sqrt(np.power(10., np.arange(dim) - (dim-1.)/2.)))
        e = mvn_entropy(mu, svar)
        for ii in range(n_repeat):
            x = np.random.multivariate_normal(mu, svar, size=n_samples)
            ksg_estimator = ksg.Entropy(x)
            ek[ii] = ksg_estimator.entropy()
        npt.assert_array_less(abs(e-ek.mean()), 2. * ek.std(), 'k-nn entropy estimate was incorrect. ' +
                                      'Dim: {}, {}, {}, {}'.format(dim, e, ek.mean(), ek))
