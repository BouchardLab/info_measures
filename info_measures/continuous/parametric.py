import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import multivariate_normal


def multivariate_normal_mutual_information(x, y):
    x = x[:, x.std(axis=0) > 0]
    y = y[:, y.std(axis=0) > 0]
    x = PCA(whiten=True).fit_transform(x)
    y = PCA(whiten=True).fit_transform(y)
    joint = np.concatenate([x, y], axis=1)
    joint_cov = np.cov(joint, rowvar=False)
    x_cov = np.cov(x, rowvar=False)
    y_cov = np.cov(y, rowvar=False)
    return (np.linalg.slogdet(x_cov)[1] + np.linalg.slogdet(y_cov)[1] -
            np.linalg.slogdet(joint_cov)[1])

class GaussianMixture(object):
    def __init__(self, n_components=1, covariance_type='full', n_init=1,
                 seed=20180423):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.rng = np.random.RandomState(seed)
        self.n_init = n_init

    def fit(self, x, y):
        self.nx = x.shape[1]
        self.ny = y.shape[1]
        self.nj = self.nx + self.ny
        pca_x = PCA(whiten=True)
        x = pca_x.fit_transform(x)
        pca_y = PCA(whiten=True)
        y = pca_y.fit_transform(y)
        self.pcas = [pca_x, pca_y]
        joint = np.concatenate([x, y], axis=1)
        self.joint_model = GMM(n_components=self.n_components,
                               covariance_type=self.covariance_type,
                               n_init=self.n_init)
        self.joint_model.fit(joint)
        return self


    def eval_p(self, X):
        logp = self.eval_logp(X)
        return np.exp(logp)


    def eval_logp(self, X, whiten=False):
        if whiten:
            x = X[:, :self.nx]
            y = X[:, self.nx:]
            x = self.pcas[0].transform(x)
            y = self.pcas[1].transform(y)
            X = np.concatenate([x, y], axis=1)
        logp = self.joint_model.score_samples(X)
        return logp


    def sample(self, n_samples):
        return self.joint_model.sample(n_samples)[0]


    def sample_entropy(self, n_samples):
        samples = self.sample(n_samples)
        return samples, -self.eval_logp(samples).mean()


    def eval_logp_xy(self, X, white=False):
        x = X[:, :self.nx]
        y = X[:, self.nx:]
        if whiten:
            x = self.pcas[0].transform(x)
            y = self.pcas[1].transform(y)
        weights = self.joint_model.weights_
        means = self.joint_model.means_
        means_x = means[:, :self.nx]
        means_y = means[:, self.nx:]
        covs = self.joint_model.covariances_
        if self.covariance_type == 'full':
            pass
        elif self.covariance_type == 'spherical':
            covs = np.stack([np.eye(self.nj)*covs[ii] for ii in range(self.n_components)])
        elif self.covariance_type == 'tied':
            covs = np.tile(covs[np.newaxis], (self.n_components, 1, 1))
        elif self.covariance_type == 'diag':
            covs = np.stack([np.diag(covs[ii]) for ii in range(self.n_components)])
        else:
            raise NotImplementedError
        covs_x = covs[:, :self.nx, :self.nx]
        covs_y = covs[:, self.nx:, self.nx:]

        logps_x = np.array([multivariate_normal(means_x[ii], covs_x[ii]).logpdf(x) for ii in range(self.n_components)]).T
        logps_y = np.array([multivariate_normal(means_y[ii], covs_y[ii]).logpdf(y) for ii in range(self.n_components)]).T

        Xlogp = logsumexp(logps_x, b=weights[np.newaxis], axis=-1)
        Ylogp = logsumexp(logps_y, b=weights[np.newaxis], axis=-1)
        return Xlogp, Ylogp


    def sample_mutual_information(self, n_samples):
        samples = self.sample(n_samples)
        logp = self.eval_logp(samples)
        Xlogp, Ylogp = self.eval_logp_xy(samples)
        return samples, (logp - Xlogp - Ylogp).mean()
