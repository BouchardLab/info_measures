import numpy as np
from scipy.special import logsumexp


class GaussianMixture(object):
    def __init__(self, n_mixtures, dim, center_var, var, seed=20180423):
        self.n_mixtures = n_mixtures
        self.dim = dim
        self.center_var = center_var
        self.var = var
        self.rng = np.random.RandomState(seed)
        self.center = self.rng.normal(scale=center_var, size=(n_mixtures, dim))


    def eval_p(self, X):
        logp = self.eval_logp(X)
        return np.exp(logp)


    def eval_logp(self, X):
        assert X.ndim == 2
        assert X.shape[1] == self.dim
        Xp = X[:, np.newaxis]
        centerp = self.center[np.newaxis]
        var = self.var
        mixture_ll = -(Xp - centerp)**2 / 2. / var - .5*np.log(2. * np.pi * var)
        mixture_ll = mixture_ll.sum(axis=-1)
        logp = logsumexp(mixture_ll, axis=-1) - np.log(self.n_mixtures)
        return logp


    def sample(self, n_samples):
        n_mixtures = self.n_mixtures
        idxs = self.rng.multinomial(1, np.ones(n_mixtures)/n_mixtures, n_samples)
        idxs = idxs.argmax(axis=-1)
        samples = self.rng.randn(n_samples, self.dim) * self.var
        samples += self.center[idxs]
        return samples


    def sample_entropy(self, n_samples):
        samples = self.sample(n_samples)
        logp = self.eval_logp(samples)
        return samples, -logp.dot(np.exp(logp)) / n_samples


    def sample_mutual_information(self, n_samples, var1_dim):
        assert var1_dim < self.dim
        samples = self.sample(n_samples)
        return
