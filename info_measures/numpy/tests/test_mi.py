import numpy as np
from numpy import testing as npt
from scipy.stats import multivariate_normal

from info_measures.numpy import (kraskov_stoegbauer_grassberger as ksg,
                                 kolchinsky_tracey as kt)


def mvn_mi(rho):
    return -0.5 * np.log(1. - rho**2)

def test_mvn_ksg_mutual_information():
    """Test Kraskov=Stoegbauer-Grassberger mutual information
    estimator on multivariate normal data."""
    mu = np.array([0., 0.])
    var = np.array([[1., 1.]])
    n_repeat = 10
    n_samples = 1000
    for rho in [.1, .5, .9]:
        cov = rho * var.T.dot(var)
        cov = np.diag(np.squeeze(var)) + cov-np.diag(np.diag(cov))
        mi = mvn_mi(rho)

        mik1 = np.zeros(n_repeat)
        mik2 = np.zeros(n_repeat)
        for ii in range(n_repeat):
            z = np.random.multivariate_normal(mu, cov, size=n_samples)
            x, y = z[:,[0]], z[:,[1]]
            ksg_estimator = ksg.MutualInformation(x, y)
            mik1[ii] = ksg_estimator.mutual_information(kind=1)
            mik2[ii] = ksg_estimator.mutual_information(kind=2)
        npt.assert_array_less(abs(mi-mik1.mean()), 2. * mik1.std(),
                              'k-nn mutual information estimate was incorrect.')
        npt.assert_array_less(abs(mi-mik2.mean()), 2. * mik2.std(),
                              'k-nn mutual information estimate was incorrect.')


"""
def test_mvn_ksg_mutual_information():
    """Test Kraskov=Stoegbauer-Grassberger mutual information
    estimator on multivariate normal data."""
    mu = np.array([0., 0.])
    var = np.array([[1., 1.]])
    n_repeat = 10
    n_samples = 1000
    for rho in [.1, .5, .9]:
        cov = rho * var.T.dot(var)
        cov = np.diag(np.squeeze(var)) + cov-np.diag(np.diag(cov))
        mi = mvn_mi(rho)

        mik1 = np.zeros(n_repeat)
        mik2 = np.zeros(n_repeat)
        for ii in range(n_repeat):
            z = np.random.multivariate_normal(mu, cov, size=n_samples)
            x, y = x[:,[0]], x[:,[1]]
            ksg_estimator = ksg.MutualInformation(x, y)
            mik1[ii] = ksg_estimator.mutual_information(kind=1)
            mik2[ii] = ksg_estimator.mutual_information(kind=2)
        npt.assert_array_less(abs(mi-mik1.mean()), 2. * mik1.std(),
                              'k-nn mutual information estimate was incorrect.')
        npt.assert_array_less(abs(mi-mik2.mean()), 2. * mik2.std(),
                              'k-nn mutual information estimate was incorrect.')
                              """
