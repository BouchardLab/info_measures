import importlib.resources

import numpy as np
from scipy.io import loadmat


"""Code modified from: https://github.com/EEthinker/JVHW_Entropy_Estimators
MATLAB and Python 2.7/3 Implementations of the JVHW (Jiao--Venkat--Han--Weissman) entropy and
mutual information estimators in Jiantao Jiao, Kartik Venkat, Yanjun Han, and Tsachy Weissman.
"Minimax estimation of functionals of discrete distributions." IEEE Transactions on Information
Theory 61, no. 5 (2015): 2835-2885.
"""


def est_entro_JVHW(X):
    """JVHW estimate of Shannon entropy (in bits) of the input samples. This function handles input
    encoding via np.unique.

    Parameters
    ----------
    X: ndarray (n_samples, ...)
        A ndarray of samples from a discrete alphabet. For 1d arrays, each element is taken as a
        samples. For Nd arrays, each row is taken as a sample.

    Returns
    -------
     est: float
         The entropy (in bits) of the input samples.
    """
    if X.ndim == 1:
        _, X = np.unique(X, return_inverse=True)
    else:
        _, X = np.unique(X, axis=0, return_inverse=True)
    return np.squeeze(_est_entro_JVHW(X))


def est_entro_MLE(X):
    """ML estimate of Shannon entropy (in bits) of the input samples. This function handles input
    encoding via np.unique.

    Parameters
    ----------
    X: ndarray (n_samples, ...)
        A ndarray of samples from a discrete alphabet. For 1d arrays, each element is taken as a
        samples. For Nd arrays, each row is taken as a sample.

    Returns
    -------
     est: float
         The entropy (in bits) of the input samples.
    """
    if X.ndim == 1:
        _, X = np.unique(X, return_inverse=True)
    else:
        _, X = np.unique(X, axis=0, return_inverse=True)
    return np.squeeze(_est_entro_MLE(X))


def est_MI_JVHW(X, Y):
    """JVHW estimate of Shannon mutual information (in bits) of the input samples. This function
    handles input encoding via np.unique.

    Parameters
    ----------
    X: ndarray (n_samples, ...)
        A ndarray of samples from a discrete alphabet. For 1d arrays, each element is taken as a
        samples. For Nd arrays, each row is taken as a sample.
    Y: ndarray (n_samples, ...)
        A ndarray of samples from a discrete alphabet. For 1d arrays, each element is taken as a
        samples. For Nd arrays, each row is taken as a sample.

    Returns
    -------
     est: float
         The mutual information (in bits) of the input samples.
    """
    if X.ndim == 1:
        _, X = np.unique(X, return_inverse=True)
        _, Y = np.unique(Y, return_inverse=True)
    else:
        _, X = np.unique(X, axis=0, return_inverse=True)
        _, Y = np.unique(Y, axis=0, return_inverse=True)
    return np.squeeze(_est_MI_JVHW(X, Y))


def est_MI_MLE(X, Y):
    """ML estimate of Shannon mutual information (in bits) of the input sample. This function
    handles input encoding via np.unique.

    Parameters
    ----------
    X: ndarray (n_samples, ...)
        A ndarray of samples from a discrete alphabet. For 1d arrays, each element is taken as a
        samples. For Nd arrays, each row is taken as a sample.
    Y: ndarray (n_samples, ...)
        A ndarray of samples from a discrete alphabet. For 1d arrays, each element is taken as a
        samples. For Nd arrays, each row is taken as a sample.

    Returns
    -------
     est: float
         The entropy (in bits) of the input samples.
    """
    if X.ndim == 1:
        _, X = np.unique(X, return_inverse=True)
        _, Y = np.unique(Y, return_inverse=True)
    else:
        _, X = np.unique(X, axis=0, return_inverse=True)
        _, Y = np.unique(Y, axis=0, return_inverse=True)
    return np.squeeze(_est_MI_MLE(X, Y))


def _est_entro_JVHW(samp):
    """Proposed JVHW estimate of Shannon entropy (in bits) of the input sample
    This function returns a scalar JVHW estimate of the entropy of samp when
    samp is a vector, or returns a row vector containing the JVHW estimate of
    each column of samp when samp is a matrix.
    Input:
    ----- samp: a vector or matrix which can only contain integers. The input
                data type can be any interger classes such as uint8/int8/
                uint16/int16/uint32/int32/uint64/int64, or floating-point
                such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each column
               of the input matrix. The output data type is double.
    """
    samp = formalize_sample(samp)
    [n, wid] = samp.shape
    n = float(n)

    # The order of polynomial is no more than 22 because otherwise floating-point error occurs
    order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
    with importlib.resources.path('info_measures.discrete', 'poly_coeff_entro.mat') as f:
        fpath = f
    poly_entro = loadmat(fpath)['poly_entro']
    coeff = poly_entro[order - 1, 0][0]

    f = fingerprint(samp)

    prob = np.arange(1, f.shape[0] + 1) / n

    # Piecewise linear/quadratic fit of c_1
    V1 = np.array([0.3303, 0.4679])
    V2 = np.array([-0.530556484842359, 1.09787328176926, 0.184831781602259])
    f1nonzero = f[0] > 0
    c_1 = np.zeros(wid)

    with np.errstate(divide='ignore', invalid='ignore'):
        if n >= order and f1nonzero.any():
            if n < 200:
                c_1[f1nonzero] = np.polyval(V1, np.log(n / f[0, f1nonzero]))
            else:
                n2f1_small = f1nonzero & (np.log(n / f[0]) <= 1.5)
                n2f1_large = f1nonzero & (np.log(n / f[0]) > 1.5)
                c_1[n2f1_small] = np.polyval(V2, np.log(n / f[0, n2f1_small]))
                c_1[n2f1_large] = np.polyval(V1, np.log(n / f[0, n2f1_large]))

            # make sure nonzero threshold is higher than 1/n
            c_1[f1nonzero] = np.maximum(c_1[f1nonzero], 1 / (1.9 * np.log(n)))

        prob_mat = entro_mat(prob, n, coeff, c_1)

    return np.sum(f * prob_mat, axis=0) / np.log(2)


def entro_mat(x, n, g_coeff, c_1):
    # g_coeff = {g0, g1, g2, ..., g_K}, K: the order of best polynomial approximation,
    K = len(g_coeff) - 1
    thres = 4 * c_1 * np.log(n) / n
    T, X = np.meshgrid(thres, x)
    ratio = np.minimum(np.maximum(2 * X / T - 1, 0), 1)
    q = np.arange(K).reshape((1, 1, K))
    g = g_coeff.reshape((1, 1, K + 1))
    MLE = - X * np.log(X) + 1 / (2 * n)
    polyApp = np.sum(np.concatenate((T[..., None], ((n * X)[..., None] - q) /
                                     (T[..., None] * (n - q))),
                                    axis=2).cumprod(axis=2) * g, axis=2) - X * np.log(T)
    polyfail = np.isnan(polyApp) | np.isinf(polyApp)
    polyApp[polyfail] = MLE[polyfail]
    output = ratio * MLE + (1 - ratio) * polyApp
    return np.maximum(output, 0)


def _est_entro_MLE(samp):
    """Maximum likelihood estimate of Shannon entropy (in bits) of the input
    sample
    This function returns a scalar MLE of the entropy of samp when samp is a
    vector, or returns a (row-) vector consisting of the MLE of the entropy
    of each column of samp when samp is a matrix.
    Input:
    ----- samp: a vector or matrix which can only contain integers. The input
                data type can be any interger classes such as uint8/int8/
                uint16/int16/uint32/int32/uint64/int64, or floating-point
                such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each
               column of the input matrix. The output data type is double.
    """
    samp = formalize_sample(samp)
    [n, wid] = samp.shape
    n = float(n)

    f = fingerprint(samp)
    prob = np.arange(1, f.shape[0] + 1) / n
    prob_mat = - prob * np.log2(prob)
    return prob_mat.dot(f)


def formalize_sample(samp):
    samp = np.array(samp)
    if np.any(samp != np.fix(samp)):
        raise ValueError('Input sample must only contain integers.')
    if samp.ndim == 1 or samp.ndim == 2 and samp.shape[0] == 1:
        samp = samp.reshape((samp.size, 1))
    return samp


def fingerprint(samp):
    """A memory-efficient algorithm for computing fingerprint when wid is
    large, e.g., wid = 100
    """
    wid = samp.shape[1]

    d = np.r_[
        np.full((1, wid), True, dtype=bool),
        np.diff(np.sort(samp, axis=0), 1, 0) != 0,
        np.full((1, wid), True, dtype=bool)
    ]

    f_col = []
    f_max = 0

    for k in range(wid):
        a = np.diff(np.flatnonzero(d[:, k]))
        a_max = a.max()
        hist, _ = np.histogram(a, bins=a_max, range=(1, a_max + 1))
        f_col.append(hist)
        if a_max > f_max:
            f_max = a_max

    return np.array([np.r_[col, [0] * (f_max - len(col))] for col in f_col]).T


def _est_MI_JVHW(X, Y):
    """This function returns our scalar estimate of mutual information I(X;Y)
    when both X and Y are vectors, and returns a row vector consisting
    of the estimate of mutual information between each corresponding column
    of X and Y when they are matrices.

    Input:
    ----- X, Y: two vectors or matrices with the same size, which can only
                contain integers.
    Output:
    ----- est: the estimate of the mutual information between input vectors
               or that between each corresponding column of the input
               matrices. The output data type is double.
    """
    [X, Y, XY] = formalize(X, Y)

    # I(X,Y) = H(X) + H(Y) - H(X,Y)
    return np.maximum(0, _est_entro_JVHW(X) + _est_entro_JVHW(Y) - _est_entro_JVHW(XY))


def _est_MI_MLE(X, Y):
    """This function returns the scalar MLE of the mutual information I(X;Y)
    when both X and Y are vectors, and returns a row vector consisting
    of the estimate of mutual information between each corresponding column
    of X and Y when they are matrices.

    Input:
    ----- X, Y: two vectors or matrices with the same size, which can only
                contain integers.

    Output:
    ----- est: the estimate of the mutual information (in bits) between input
               vectors or that between each corresponding column of the input
               matrices. The output data type is double.
    """

    [X, Y, XY] = formalize(X, Y)

    # I(X,Y) = H(X) + H(Y) - H(X,Y)
    return np.maximum(0, _est_entro_MLE(X) + _est_entro_MLE(Y) - _est_entro_MLE(XY))


def formalize(X, Y):
    X = formalize_sample(X)
    Y = formalize_sample(Y)

    if X.shape != Y.shape:
        raise ValueError('Input arguments X and Y should be of the same size.')

    X = X.astype(np.int64, copy=False)
    Y = Y.astype(np.int64, copy=False)

    X = map_int(X)
    Y = map_int(Y)
    XY = (X - 1) * Y.max(axis=0) + Y

    return X, Y, XY


def map_int(samp):
    """Map integer data along each column of X and Y to consecutive integer
    numbers (which start with 1 and end with the total number of distinct
    values in each corresponding column). For example,
                    [  1    6    4  ]        [ 1  3  3 ]
                    [  2    6    3  ] -----> [ 2  3  2 ]
                    [  3    2    2  ]        [ 3  1  1 ]
                    [ 1e5   3   100 ]        [ 4  2  4 ]
    The purpose of this data mapping is to make the effective data range
    as small as possible, minimizing the possibility of overflows.
    """
    id = samp.argsort(axis=0)
    col_index = np.indices(samp.shape)[1]
    samp = samp[id, col_index]
    samp[id, col_index] = np.cumsum(np.r_[np.ones((1, samp.shape[1])),
                                          np.diff(samp, axis=0) > 0], axis=0)
    return samp
