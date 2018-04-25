import numpy as np
from numpy import testing as npt
from scipy.stats import (multivariate_normal, ttest_1samp)

from science_nets.numpy import (kraskov_stoegbauer_grassberger as ksg,
                                kolchinsky_tracey as kt)


def mvn_entropy(mean, covariance):
    return np.asscalar(multivariate_normal(mean, covariance).entropy())


def test_mvn_entropy():
    """Test entropy estimators on multivariate normal data."""
    n_repeat = 100
    n_samples = 10000
    eu = np.zeros(n_repeat)
    el = np.zeros(n_repeat)
    ek = np.zeros(n_repeat)
    for dim in [1, 2, 3]:
        n_samples = 500 * 10**(dim // 2)
        mu = np.zeros(dim)
        svar = np.diag(np.sqrt(np.power(10., np.arange(dim) - (dim-1.)/2.)))
        e = mvn_entropy(mu, svar)
        npt.assert_array_less(0., e)
        for ii in range(n_repeat):
            x = np.random.multivariate_normal(mu, svar, size=n_samples)
            kt_estimator = kt.Entropy(x)
            ksg_estimator = ksg.Entropy(x)
            eu[ii] = kt_estimator.entropy(lower=False)
            el[ii] = kt_estimator.entropy(lower=True)
            ek[ii] = ksg_estimator.entropy()
        npt.assert_array_less(e, eu, 'Entropy upper bound was not larger ' +
                                     'than ground truth. Dim: {}'.format(dim))
        npt.assert_array_less(0., el, 'Entropy lower bound was not larger ' +
                                      'than zero. Dim: {}'.format(dim))
        npt.assert_array_less(el, e, 'Entropy lower bound was not smaller ' +
                                     'than ground truth. Dim: {}'.format(dim))
        npt.assert_array_less(abs(e-ek.mean()), ek.std(), 'k-nn entropy estimate was incorrect. ' +
                                      'Dim: {}, {}, {}, {}'.format(dim, e, ek.mean(), ek))
