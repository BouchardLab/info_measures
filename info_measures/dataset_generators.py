import numpy as np
from scipy.special import factorial
import numba

class DatasetGenerator(object):
    """Generate datasets for use in Predictive Information calculations.
    Data is assumed to have

    Parameters
    ----------
    X : ndarray (n_batch, ..., n_features)
        Dataset.
    symmetry_axes : int or list
        List of axes with translational symmetry.
        """

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


class VectorSpaceGenerator(DatasetGenerator):
    """Generate a dataset assuming the data lives in a vectorspace."""

    def __init__(self, X):
        super(VectorSpaceGenerator, self).__init__(X)
        std = self.X.std(axis=(0, 2))
        self.X = self.X[:, std > 0]


    def sample_data(self, grow_dim, resamplings_per_datapoint=None, n_samples=None):
        n_X, n_i, n_f = self.X.shape
        """
        all_samples = n_X * factorial(n_i, exact=True) // (factorial(n_i - 2 * grow_dim, exact=True))
        log_all_samples = np.log10(float(all_samples))
        """
        n_sample_locs = n_X
        if n_samples is None:
            if resamplings_per_datapoint is None:
                resamplings_per_datapoint = 1
            n_samples = n_X * resamplings_per_datapoint
        else:
            if resamplings_per_datapoint is not None:
                raise ValueError
        iters = int(np.ceil(n_samples / n_sample_locs))

        sample_dim = 2 * grow_dim
        out = np.full((n_samples, sample_dim, n_f), np.nan)

        for ii in range(iters):
            if (ii+1)*n_X <= n_samples:
                x0 = np.tile(np.arange(n_X)[:,np.newaxis], (1, n_i))[:, :sample_dim]
                x1 = np.argsort(self.rng.randn(*self.X.shape[:2]), axis=1)[:, :sample_dim]
                X = self.X[x0, x1].reshape(n_X, sample_dim, n_f)
                out[ii*n_X:(ii+1)*n_X] = X
            else:
                for jj in range(ii*n_X, n_samples):
                    idx = self.rng.randint(n_X)
                    order = self.rng.permutation(n_i)[:sample_dim]
                    out[jj] = self.X[idx][order]
        return out


class ImageGenerator(DatasetGenerator):
    """Generate a dataset assuming the data lives in an image space.

    Parameters
    ----------
    X : ndarray (n_batch, height, width, n_channels)
        Dataset.
    """

    def __init__(self, X, grow_axis, symmetry_axes=None):
        if X.ndim != 4:
            raise ValueError('Images must be 3 dimensional: h by w by c.')
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


    def sample_data(self, grow_dim, perp_dim=None, n_samples=None):
        if perp_dim is None:
            perp_dim = self.X.shape[self.perp_axis]
        n_X, n_g, n_p, n_f = self.X.shape

        n_sample_locs = n_X * (n_g + 1 - (2 * grow_dim)) * (n_p + 1 - perp_dim)
        if n_samples is None:
            n_samples = n_sample_locs
        else:
            if n_samples > n_sample_locs:
                raise ValueError('It is not possible to generate this many unique samples.')

        sample_dim = (2 * grow_dim) * perp_dim
        out = np.full((n_samples, sample_dim, n_f), np.nan)
        if n_samples == n_sample_locs:
            print('here')
            loc = 0
            for ii in range(n_g + 1 - (2 * grow_dim)):
                for jj in range(n_p + 1 - perp_dim):
                    s = loc * n_X
                    e = (loc + 1) * n_X
                    out[s:e] = self.X[:,ii:ii+2*grow_dim,jj:jj+perp_dim].reshape(n_X, -1, n_f)
                    loc += 1
        else:
            for ii in range(n_samples):
                idx = self.rng.randint(n_X)
                g_idx = self.rng.randint(n_g - (2 * grow_dim))
                p_idx = self.rng.randint(n_p - perp_dim)
                out[ii] = self.X[idx,g_idx:g_idx+2*grow_dim,p_idx:p_idx+perp_dim].reshape(-1, n_f)
        return out


class MultiChannelTimeseriesGenerator(DatasetGenerator):
    """Generate a dataset assuming the data is multichannel timeseries.

    Many neural datasets fit this description.

    Parameters
    ----------
    X : ndarray (n_batch, n_time, n_channels, n_features)
        Dataset.
    """

    def __init__(self, X):
        symmetry_axes = 1
        super(MultiChannelTimeseriesGenerator, self).__init__(X, symmetry_axes=symmetry_axes)


    def sample_data(self, time_dim, n_channels=None, channel_resamplings_per_datapoint=None, n_samples=None):
        if n_channels is None:
            n_channels = self.X.shape[2]
        n_X, n_t, n_c, n_f = self.X.shape
        n_sample_locs = n_X * (n_t + 1 - (2 * time_dim))
        if n_samples is None:
            if channel_resamplings_per_datapoint is None:
                channel_resamplings_per_datapoint = 1
            n_samples = n_X * (n_t + 1 - (2 * time_dim)) * channel_resamplings_per_datapoint
        else:
            if channel_resamplings_per_datapoint is not None:
                raise ValueError
        iters = int(np.ceil(n_samples / n_sample_locs))
        sample_dim = (2 * time_dim) * n_channels
        out = np.full((n_samples, sample_dim, n_f), np.nan)
        loc = 0
        e = 0
        for ii in range(iters):
            if e + n_sample_locs <= n_samples:
                for jj in range(n_t + 1 - (2 * time_dim)):
                    s = loc * n_X
                    e = (loc + 1) * n_X
                    channels = self.rng.permutation(n_c)[:n_channels]
                    out[s:e] = self.X[:,jj:jj+2*time_dim][:, :, channels].reshape(n_X, -1, n_f)
                    loc += 1
            else:
                for jj in range(e, n_samples):
                    idx = self.rng.randint(n_X)
                    t = self.rng.randint(n_t - (2 * time_dim))
                    channels = self.rng.permutation(n_c)[:n_channels]
                    out[jj] = self.X[idx, t:t+2*time_dim][:, channels].reshape(-1, n_f)
        return out
