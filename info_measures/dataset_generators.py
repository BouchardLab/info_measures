import numpy as np
from scipy.special import factorial
import numba
from numba import prange, njit

class DatasetGenerator(object):
    """Generate datasets for use in Predictive Information calculations.

    Parameters
    ----------
    X : ndarray (n_batch, ..., n_features)
        Dataset.
    symmetry_axes : int or list
        List of axes with translational symmetry."""

    def __init__(self, X, symmetry_axes=None, seed=20180516):
        self.rng = np.random.RandomState(seed)
        if X.ndim < 3:
            raise ValueError("X must have at least 3 dimensions.")
        if isinstance(symmetry_axes, int):
            symmetry_axes = [symmetry_axes]
        if symmetry_axes is not None:
            if len(symmetry_axes) >= X.ndim-1:
                raise ValueError("symmetry_axes can be at most X.ndim-2.")
            axes_order = [0]
            axes_order.extend(symmetry_axes)
            for axis in range(1, X.ndim):
                if axis not in symmetry_axes:
                    axes_order.append(axis)
            X = np.transpose(X, axes=(axes_order))
            sh = X.shape
            new_shape = [sh[0]]
            new_shape.extend([sh[ii] for ii in range(1, len(symmetry_axes)+1)])
            if np.prod(axes_order[1+len(symmetry_axes):-1]) > 1:
                new_shape.append(-1)
            new_shape.append(X.shape[-1])
            X = X.reshape(new_shape)
            symmetry_axes = list(range(1, len(symmetry_axes)+1))
        else:
            X = X.reshape(X.shape[0], -1, X.shape[-1])
        self.X = X
        self.symmetry_axes = symmetry_axes


        def sample_data(self, grow_dim, sample_all=True):
            raise NotImplementedError

@njit
def _sample_data2(X, out, grow_dim, n_samples_per_image):
    n_X, n_i, n_f = X.shape
    sample_dim = 2 * grow_dim
    for ii in range(n_samples_per_image):
        for jj in range(n_X):
            out[ii*n_X + jj] = np.random.permutation(X[jj])[:sample_dim]


@njit(parallel=True)
def _sample_data3(X, out, grow_dim, n_samples_per_image):
    n_X, n_i, n_f = X.shape
    sample_dim = 2 * grow_dim
    for ii in prange(n_samples_per_image*n_X):
        jj = ii // n_X
        out[ii] = np.random.permutation(X[jj])[:sample_dim]


class VectorSpaceGenerator(DatasetGenerator):
    """Generate a dataset assuming the data lives in a vectorspace."""

    def __init__(self, X):
        super(VectorSpaceGenerator, self).__init__(X)
        std = self.X.std(axis=(0, 2))
        self.X = self.X[:, std > 0]


    def sample_data(self, grow_dim, n_samples_per_image):
        n_X, n_i, n_f = self.X.shape
        X = np.transpose(self.X, (1, 0, 2)).copy()
        """
        all_samples = n_X * factorial(n_i, exact=True) // (factorial(n_i - 2 * grow_dim, exact=True))
        log_all_samples = np.log10(float(all_samples))
        """
        n_samples = n_X * n_samples_per_image
        sample_dim = 2 * grow_dim
        out = np.full((sample_dim, n_samples, n_f), np.nan)
        for ii in range(n_samples_per_image):
            self.rng.shuffle(X)
            out[:,ii*n_X:(ii+1)*n_X] = X[:sample_dim]
        return np.transpose(out, (1, 0, 2))


    def sample_data2(self, grow_dim, n_samples_per_image):
        n_X, n_i, n_f = self.X.shape
        """
        all_samples = n_X * factorial(n_i, exact=True) // (factorial(n_i - 2 * grow_dim, exact=True))
        log_all_samples = np.log10(float(all_samples))
        """
        n_samples = n_X * n_samples_per_image
        sample_dim = 2 * grow_dim
        out = np.full((n_samples, sample_dim, n_f), np.nan)
        _sample_data2(self.X, out, grow_dim, n_samples_per_image)
        return out

    def sample_data3(self, grow_dim, n_samples_per_image):
        n_X, n_i, n_f = self.X.shape
        """
        all_samples = n_X * factorial(n_i, exact=True) // (factorial(n_i - 2 * grow_dim, exact=True))
        log_all_samples = np.log10(float(all_samples))
        """
        n_samples = n_X * n_samples_per_image
        sample_dim = 2 * grow_dim
        out = np.full((n_samples, sample_dim, n_f), np.nan)
        _sample_data3(self.X, out, grow_dim, n_samples_per_image)
        return out


class ImageGenerator(DatasetGenerator):
    """Generate a dataset assuming the data lives in a vectorspace."""

    def __init__(self, X, grow_axis, symmetry_axes=None):
        if symmetry_axes is None:
            symmetry_axes = (1, 2)
        if len(symmetry_axes) != 2:
            raise ValueError('Images must have two symmetry_axes')
        if grow_axis not in symmetry_axes:
            raise ValueError
        self.grow_axis = 1
        self.perp_axis = 2
        if grow_axis == symmetry_axes[1]:
            symmetry_axes = (symmetry_axes[1], symmetry_axes[0])
        super(ImageGenerator, self).__init__(X, symmetry_axes=symmetry_axes)


    def sample_data(self, grow_dim, perp_dim = None, sample_all=True):
        if perp_dim is None:
            perp_dim = self.X.shape[self.perp_axis]

        n_X, n_g, n_p, n_f = self.X.shape
        n_samples = n_X * (n_g + 1 - 2 * grow_dim) * (n_p + 1 - perp_dim)
        sample_dim = (2 * grow_dim) * perp_dim * np.prod(self.X.shape[3:], dtype=int)
        out = np.full((n_samples, sample_dim, n_f), np.nan)
        loc = 0
        for ii in range(n_g + 1 - (2 * grow_dim)):
            for jj in range(n_p + 1 - perp_dim):
                s = loc * n_X
                e = (loc + 1) * n_X
                out[s:e] = self.X[:,ii:ii+2*grow_dim,jj:jj+perp_dim].reshape(n_X, -1, n_f)
                loc += 1
        return out
