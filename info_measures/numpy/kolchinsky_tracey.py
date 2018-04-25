import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


"""
Adapted from https://github.com/artemyk/ibsgd
"""


class Entropy(object):
    """Estimate the entropy of a distribution based on a kernel-density
    estimate.

    Parameters
    ----------
    X : data (n_samples, n_dim)
        Data matrix
    var : float
        Variance for Normal mixtures. Can either be set on instantiation
        or when entropy is computed.
    lower : bool (default True)
        Whether to estimate a lower or upper bound.
    """
    def __init__(self, X, var=1e-2, lower=True):
        self.pairwise_dists = self.pairwise_distance(X)
        self.lower = lower
        self.var = var
        self.n_samples, self.n_dim = X.shape


    @staticmethod
    def pairwise_distance(X):
        X_sqr = (X * X).sum(axis=-1, keepdims=True)
        dists = X_sqr + X_sqr.T - 2. * X.dot(X.T)
        return dists


    def kde_condentropy(self, var=None):
        # Return entropy of a multivariate Gaussian, in nats
        if var is None:
            if self.var is None:
                raise ValueError('Mixture variance (var) not specified.')
            else:
                var = self.var
        return self.n_dim * (np.log(2. * np.pi * var) + 1.) / 2.


    def entropy_estimator_kl(self, var):
        # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
        #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
        #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
        mvn_lp = self.pairwise_dists / (-2. * var)
        lse = logsumexp(mvn_lp, axis=1)
        h = self.n_dim / 2. + np.log(self.n_samples) + self.n_dim * np.log(2. * np.pi * var) / 2. - lse.mean()
        return h


    def entropy_estimator_bd(self, var):
        # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I
        #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
        h = self.n_dim * np.log(0.25) / 2. + self.entropy_estimator_kl(4. * var)
        return h

    def entropy(self, var=None, lower=None):
        if var is None:
            if self.var is None:
                raise ValueError('Mixture variance (var) not specified.')
            else:
                var = self.var
        if lower is None:
            lower = self.lower

        if lower:
            return self.entropy_estimator_bd(var)
        else:
            return self.entropy_estimator_kl(var)


class MutualInformation(object):
    """Estimate the mutual information between two variables based on a
    kernel-density estimate.

    Parameters
    ----------
    X : data (n_samples, n_dim)
        Data matrix
    var : float
        Variance for Normal mixtures. Can either be set on instantiation
        or when entropy is computed.
    lower : bool (default True)
        Whether to estimate a lower or upper bound.
    """
    def __init__(self, X, Y, var=1e-2, lower=True, symmetric=False):
        n_Xsamples, self.n_Xdim = X.shape
        n_Ysamples, self.n_Ydim = Y.shape
        assert n_Xsamples == n_Ysamples
        self.lower = lower
        self.var = var
        self.symmetric = symmetric
        self.XEnt = Entropy(X, var=var, lower=lower)
        self.YEnt = Entropy(X, var=var, lower=lower)
        self.ZEnt = Entropy(np.concatenate([X, Y], axis=1), var=var, lower=lower)


    def mutual_information(self, var=None, lower=None, symmetric=None):
        if var is None:
            if self.var is None:
                raise ValueError('Mixture variance (var) not specified.')
            else:
                var = self.var
        if lower is None:
            lower = self.lower
            """
        if symmetric is None:
            symmetric = self.symmetric
        mi = self.XEnt.entropy(var, lower) - self.XEnt.kde_condentropy(var)
        if symmetric:
            mi = (mi + self.YEnt.entropy(var, lower) -
                  self.YEnt.kde_condentropy(var)) / 2.
                  """
        mi = (self.XEnt.entropy(var, lower) + self.YEnt.entropy(var, lower) -
              self.ZEnt.entropy(var, lower))
        return mi
