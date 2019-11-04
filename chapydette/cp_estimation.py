# coding=utf-8
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import scipy.spatial
import sklearn.metrics

import chapydette.compute_gram as compute_gram
import chapydette.mkcpe as run_mkcpe
import chapydette.mmd_cpd as run_mmd_cpd


def mkcpe(X=None, gram=None, n_cp=1, kernel_type='detect', bw=None, min_dist=1, return_obj=False):
    """
    Run the kernel change-point algorithm of Harchaoui and Cappé (2007) to detect a fixed number n_cp of change points
    in a sequence of observations.

    Reference:

    * Harchaoui, Z., & Cappé, O. (2007, August). Retrospective multiple change-point estimation with kernels. In IEEE/SP 14th Workshop on Statistical Signal Processing, 2007. SSP'07. (pp. 768-772). IEEE.

    :param X: Matrix of observations. Each observation is one row.
    :type X: numpy.ndarray
    :param gram: Pre-computed gram matrix
    :type gram: numpy.ndarray
    :param n_cp: Number of change points to detect. Either an integer or tuple of integers containing the min and max of
                an interval for the # change points to detect.
    :type n_cp: int or array_like
    :param kernel_type: Type of kernel to use. One of: 'detect', 'precomputed', 'chi-squared', 'gaussian-euclidean',
                        'gaussian-hellinger', 'gaussian-tv', 'linear'. If 'detect', it chooses the Gaussian kernel with
                        the Hellinger distance if the data consists of histograms and the linear kernel otherwise.
    :type kernel_type: str
    :param bw: Bandwidth for the kernel (if applicable)
    :type bw: float
    :param min_dist: Minimum allowable distance between successive change points (in terms of indices)
    :type min_dist: int
    :param return_obj: Whether to return the optimal objective value
    :type return_obj: bool
    :return: cps: Indices of the estimated change points (the indices of the last element in each estimated segment)
    :rtype: numpy.ndarray or dict of numpy.ndarrays
    """

    kernel_num, bw, X = setup(X, gram, n_cp, kernel_type, bw, min_dist)

    if not isinstance(n_cp, collections.Sequence):
        if gram is not None:
            cps, obj_val = run_mkcpe.kcpe(gram, n_cp+1, kernel_num, bw, min_dist)  # The last value returned is the last observation
        else:
            cps, obj_val = run_mkcpe.kcpe(X, n_cp+1, kernel_num, bw, min_dist)
        cps = cps[:-1]
    else:
        if gram is not None:
            cps, obj_val = run_mkcpe.kcpe_range_cp(gram, n_cp[0]+1, n_cp[1]+1, kernel_num, bw, min_dist)  # The last value returned is the last observation
        else:
            cps, obj_val = run_mkcpe.kcpe_range_cp(X, n_cp[0]+1, n_cp[1]+1, kernel_num, bw, min_dist)
        for key in cps.keys():
            cps[key] = cps[key][:-1]

    if not return_obj:
        return cps
    else:
        return cps, obj_val


def mmd_cpd(X=None, gram=None, n_cp=1, kernel_type='detect', bw=None, min_dist=1):
    """
    Run a change point algorithm based on the Maximum Mean Discrepancy to detect a single change point

    References:

    * Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. Journal of Machine Learning Research, 13(Mar), 723-773.
    * Harchaoui, Z., Moulines, E., & Bach, F. R. (2009). Kernel change-point analysis. In Advances in Neural Information Processing Systems (pp. 609-616).

    :param X: Matrix of observations. Each observation is one row.
    :type X: numpy.ndarray
    :param gram: Pre-computed gram matrix
    :type gram: numpy.ndarray
    :param n_cp: Number of change points to detect
    :type n_cp: int
    :param kernel_type: Type of kernel to use. One of: 'detect', 'precomputed', 'chi-squared', 'gaussian-euclidean',
                        'gaussian-hellinger', 'gaussian-tv', 'linear'. If 'detect', it chooses the Gaussian kernel with
                        the Hellinger distance if the data consists of histograms and the linear kernel otherwise.
    :type kernel_type: str
    :param bw: Bandwidth for the kernel (if applicable)
    :type bw: float
    :param min_dist: Minimum allowable distance between successive change points (in terms of indices)
    :type min_dist: int
    :return: cp: Index of the estimated change point (the index of the last element in the first segment)
    :rtype: int
    """
    if n_cp != 1:
        raise NotImplementedError('The method has only been implemented to detect one change point.')

    kernel_num, bw, X = setup(X, gram, n_cp, kernel_type, bw, min_dist)
    if gram is None:
        gram = compute_gram.compute_gram(X, kernel_num, bw)
    cp, obj_vals = run_mmd_cpd.mmd_cpd(gram)

    return cp


def setup(X, gram, n_cp, kernel_type, bw, min_dist):
    """
    Perform input checking, assign a value to the specified kernel, and compute the bandwidth using the rule of thumb if
    the bandwidth wasn't specified.
    :param X: Matrix of observations. Each observation is one row.
    :param gram: Pre-computed gram matrix
    :param n_cp: Number of change points to detect
    :param kernel_type: Type of kernel to use. One of: 'detect', 'precomputed', 'chi-squared', 'gaussian-euclidean',
                        'gaussian-hellinger', 'gaussian-tv', 'linear'. If 'detect', it chooses the Gaussian kernel with
                        the Hellinger distance if the data consists of histograms and the linear kernel otherwise.
    :param bw: Bandwidth for the kernel (if applicable)
    :param min_dist: Minimum allowable distance between successive change points
    :return: kernel_num: Value assigned to the specified kernel
             bw: Bandwidth to be used
    """
    if kernel_type == 'detect':
        row_sums = np.sum(X, axis=1)
        if np.isclose(np.min(row_sums), 1) and np.isclose(np.max(row_sums), 1):
            kernel_type = 'gaussian-hellinger'
            print('Using the Gaussian kernel with the Hellinger distance.')
        else:
            kernel_type = 'linear'
            print('Using the linear kernel.')

    if kernel_type.lower() == 'precomputed':
        kernel_num = -1
    elif kernel_type.lower() == 'chi-squared':
        kernel_num = 0
    elif kernel_type.lower() == 'gaussian-euclidean':
        kernel_num = 1
    elif kernel_type.lower() == 'gaussian-hellinger':
        # kernel_num = 2
        X = np.sqrt(X)
        kernel_num = 1
    elif kernel_type.lower() == 'gaussian-tv':
        kernel_num = 3
    elif kernel_type.lower() == 'linear':
        kernel_num = 4
    else:
        raise ValueError("kernel_type must be one of 'precomputed', 'chi-squared', 'gaussian-euclidean', "
                         "'gaussian-hellinger', 'gaussian-tv', or 'linear'")

    if gram is None and X is None:
        raise ValueError('You must provide either the matrix of observations or a precomputed gram matrix.')
    if gram is not None and X is not None:
        print('Using the provided gram matrix and ignoring the matrix of observations.')

    if isinstance(n_cp, int):
        if n_cp <= 0:
            raise ValueError('Number of change points n_cp must be a positive integer.')
        max_cp = n_cp
    elif isinstance(n_cp, collections.Sequence):
        if not isinstance(n_cp[0], int) or not isinstance(n_cp[1], int):
            raise ValueError('Number of change points n_cp[0] and n_cp[1] must be positive integers.')
        elif n_cp[1] < n_cp[0]:
            raise ValueError('n_cp[1] < n_cp[0] is not allowed.')
        max_cp = n_cp[1]
    else:
        raise ValueError('Number of change points n_cp must be a positive integer.')

    if min_dist <= 0 or not isinstance(min_dist, int):
        raise ValueError('Minimum allowable distance min_dist, must be a positive integer.')
    if (X is not None and max_cp > len(X)) or (gram is not None and max_cp > len(gram)):
        raise ValueError('Number of expected segments cannot exceed the size of the data vector.')
    if gram is None and bw is None and kernel_type.lower() not in ['precomputed', 'linear', 'chi-squared']:
        print('Bandwidth not specified. Using a rule of thumb.')
        if kernel_type.lower() == 'chi-squared':
            pass
        elif kernel_type.lower() == 'gaussian-euclidean':
            dists = sklearn.metrics.pairwise.pairwise_distances(X).reshape(-1)
        elif kernel_type.lower() == 'gaussian-hellinger':
            dists = sklearn.metrics.pairwise.pairwise_distances(np.sqrt(X)).reshape(-1)
        elif kernel_type.lower() == 'gaussian-tv':
            dists = scipy.spatial.distance.cdist(X, X, 'cityblock')
        bw = np.median(dists)
    elif gram is None and bw is None and kernel_type.lower() in ['chi-squared']:
        raise ValueError('You must specify a bandwidth for the chi-squared kernel.')

    if bw is None:
        bw = 1.0  # won't be used, but need to pass a value for mkcpe

    return kernel_num, bw, X