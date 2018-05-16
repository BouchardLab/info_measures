import numpy as np

class DatasetGenerator(object):
    """Generate datasets for use in Predictive Information calculations.

    Parameters
    ----------
    X : ndarray (n_batch, ...)
        Dataset.
    symmetry_axes : int or list
        List of axes with translational symmetry."""

    def __init__(self, X, symmetry_axes=None):
        if X.ndim < 2:
            raise ValueError("X must have at least 2 dimensions.")
        if isinstance(symmetry_axes, int):
            symmetry_axes = [symmetry_axes]
        if symmetry_axes is not None:
            if len(symmetry_axes) >= X.ndim:
                raise ValueError("symmetry_axes can be at most X.ndim-1.")
            axes_order = [0]
            axes_order.extend(symmetry_axes)
            for axis in range(1, X.ndim):
                if axis not in symmetry_axes:
                    axes_order.append(axis)
            X = np.transpose(X, axes=(axes_order))
            sh = X.shape
            new_shape = [sh[0]]
            new_shape.extend([sh[ii] for ii in range(1, len(symmetry_axes)+1)])
            new_shape.append(-1)
            X = X.reshape(new_shape)
            symmetry_axes = list(range(1, len(symmetry_axes)+1))
        else:
            X = X.reshape(X.shape[0], -1)
        self.X = X
        self.symmetry_axes = symmetry_axes


        def sample_data(self, dim, sample_all=True):
            raise NotImplementedError


class VectorSpaceGenerator(DatasetGenerator):
    """Generate a dataset assuming the data lives in a vectorspace."""

    def __init__(self, X):
        super(VectorSpaceGenerator, self).__init__(X)


    def sample_data(self, dim, sample_all=True):
        raise NotImplementedError


class VectorSpaceGenerator(DatasetGenerator):
    """Generate a dataset assuming the data lives in a vectorspace."""

    def __init__(self, X):
        super(VectorSpaceGenerator, self).__init__(X)


    def sample_data(self, dim, sample_all=True):
        raise NotImplementedError
