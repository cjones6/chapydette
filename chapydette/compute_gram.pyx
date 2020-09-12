# coding=utf-8
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, fabs, sqrt


cpdef compute_gram(np.ndarray[np.float64_t, ndim=2] X, int kernel_type, double bw):
    """
    Compute the gram matrix for data matrix X using a kernel with bandwidth bw
    :param X: Data matrix
    :param kernel_type: Type of kernel to use:
                        0: Chi-square
                        1: Gaussian with Euclidean distance
                        2: Gaussian with Hellinger distance
                        3: Gaussian with TV distance
                        4: Linear 
    :param bw: Bandwidth for kernel
    """
    cdef unsigned int n = np.size(X, 0)
    cdef unsigned int d = np.size(X, 1)
    cdef np.ndarray gram_matrix = np.zeros(n*n, dtype=np.float64)
    cdef double[::1] X_view = X.flatten()

    compute_gram_c(X_view, <double*> gram_matrix.data, n, d, kernel_type, bw)

    return gram_matrix.reshape((n, n))


cdef void compute_gram_c(double[::1] X, double *gram, unsigned int n, unsigned int d, int kernel_type, double bw):
    """
    Compute the gram matrix for data matrix X using a kernel with bandwidth bw
    :param X: Data matrix
    :param gram: Gram matrix (to be filled in)
    :param n: Total number of samples
    :param d: Dimension of each sample
    :param kernel_type: Type of kernel to use:
                        0: Chi-square
                        1: Gaussian with Euclidean distance
                        2: Gaussian with Hellinger distance
                        3: Gaussian with TV distance
                        4: Linear
    :param bw: Bandwidth for kernel
    """
    cdef int i, j

    for i in range(0, n):
        for j in range(i, n):
            gram[i+j*n] = compute_gram_entry(X, i, j, n, d, kernel_type, bw)
            gram[j+i*n] = gram[i+j*n]


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double compute_gram_entry(double[::1] X, long i, long j, long n, unsigned int d, int kernel_type, double bw) nogil:
    """
    Compute the value of a kernel with bandwidth bw between samples i and j from data matrix X
    :param X: Data matrix
    :param i: First sample to use when computing kernel entry
    :param j: Second sample to use when computing kernel entry
    :param n: Total number of samples
    :param d: Dimension of each sample
    :param kernel_type: Type of kernel to use:
                        0: Chi-square 
                        1: Gaussian with Euclidean distance
                        2: Gaussian with Hellinger distance
                        3: Gaussian with TV distance
                        4: Linear 
    :param bw: Bandwidth for kernel
    :return: entry: Kernel value
    """

    cdef int m
    cdef double entry = 0.0

    for m in range(d):
        if kernel_type == 0:
            if X[m+d*i]+X[m+d*j] != 0:
                entry += (X[m+d*i]-X[m+d*j])*(X[m+d*i]-X[m+d*j])/(X[m+d*i]+X[m+d*j])
        elif kernel_type == 1:
            entry += (X[m+d*i]-X[m+d*j])*(X[m+d*i]-X[m+d*j])
        elif kernel_type == 2:
            entry += (sqrt(X[m+d*i])-sqrt(X[m+d*j]))*(sqrt(X[m+d*i])-sqrt(X[m+d*j]))
        elif kernel_type == 3:
            entry += fabs(X[m+d*i]-X[m+d*j])
        elif kernel_type == 4:
            entry += X[m+d*i]*X[m+d*j]

    if kernel_type == 3:
        entry = entry*entry/4.0
    if 0 <= kernel_type <= 3:
        entry *= -1/(2.0*bw*bw)
        entry = exp(entry)

    return entry
