# coding=utf-8
from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
from openmp cimport omp_get_max_threads

from cpython.mem cimport PyMem_Malloc, PyMem_Free

from chapydette.compute_gram cimport compute_gram_entry


cdef inline long long_max(long a, long b) nogil: return a if a >= b else b
cdef inline long long_min(long a, long b) nogil: return a if a <= b else b

@cython.boundscheck(False)
cpdef kcpe(np.ndarray[np.float64_t, ndim=2] X, int k, int kernel_type, double bw, unsigned int min_dist):
    """
    Run the kernel changepoint algorithm of Harchaoui and Cappé (2007) to detect a fixed number k of changepoints in
    a sequence of observations X. Each row of X is one observation, unless kernel_type is specified to be -1, in which
    case X is assumed to be a pre-computed gram matrix between the observations.
    
    Reference:
    Harchaoui, Z., & Cappé, O. (2007, August). Retrospective multiple change-point estimation with kernels. 
    In  IEEE/SP 14th Workshop on Statistical Signal Processing, 2007. SSP'07. (pp. 768-772). IEEE.
    
    :param X: Matrix of observations or pre-computed gram matrix
    :param k: Number of changepoints to detect
    :param kernel_type: Type of kernel to use:
                        -1: Precomputed gram stored in X
                        0: Chi-square 
                        1: Gaussian with Euclidean distance
                        2: Gaussian with Hellinger distance
                        3: Gaussian with TV distance
                        4: Linear 
    :param bw: Bandwidth for the kernel
    :param min_dist: Minimum allowable distance between successive changepoints
    :return: T: Indices of the estimated changepoints (estimated endpoint of each segment)
             N[n - min_dist, k - 1]: The value at the optimum of the (non-penalized) objective function
    """
    cdef unsigned int n = np.size(X, 0)

    N, I = kcpe_forward_pass(X, k, k, kernel_type, bw, min_dist)
    T, obj = kcpe_backward_pass(N, I, k, n, min_dist)

    return np.asarray(T), np.float(obj)


@cython.boundscheck(False)
cpdef kcpe_range_cp(np.ndarray[np.float64_t, ndim=2] X, int min_cp, int max_cp, int kernel_type, double bw, unsigned int min_dist):
    """
    Run the kernel changepoint algorithm of Harchaoui and Cappé (2007) to detect changepoints in a sequence of 
    observations X. Each row of X is one observation, unless kernel_type is specified to be -1, in which
    case X is assumed to be a pre-computed gram matrix between the observations. The algorithm is run for a varying
    number of changepoints between min_cp and max_cp (inclusive).
    
    Reference:
    Harchaoui, Z., & Cappé, O. (2007, August). Retrospective multiple change-point estimation with kernels. 
    In  IEEE/SP 14th Workshop on Statistical Signal Processing, 2007. SSP'07. (pp. 768-772). IEEE.
    
    :param X: Matrix of observations or pre-computed gram matrix
    :param min_cp: Lower bound on interval of the number of changepoints to detect
    :param max_cp: Upper bound on interval of the number of changepoints to detect
    :param kernel_type: Type of kernel to use:
                        -1: Precomputed gram stored in X
                        0: Chi-square 
                        1: Gaussian with Euclidean distance
                        2: Gaussian with Hellinger distance
                        3: Gaussian with TV distance
                        4: Linear 
    :param bw: Bandwidth for the kernel
    :param min_dist: Minimum allowable distance between successive changepoints
    :return: all_ts: Dictionary with the estimated changepoints (estimated endpoint of each segment). The keys are the 
                     number of changepoints.
             all_objs: Dictionary with the values of the minimized (non-penalized) objective function. The keys are the 
                       number of changepoints.
    """
    cdef unsigned int n = np.size(X, 0)

    N, I = kcpe_forward_pass(X, min_cp, max_cp, kernel_type, bw, min_dist)

    all_ts = {}
    all_objs = {}
    for k in range(min_cp, max_cp+1):
        T, obj = kcpe_backward_pass(N, I, k, n, min_dist)
        all_ts[k-1] = np.asarray(T)
        all_objs[k-1] = np.float(obj)

    return all_ts, all_objs


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef kcpe_forward_pass(np.ndarray[np.float64_t, ndim=2] X, int min_cp, int max_cp, int kernel_type, double bw, unsigned int min_dist):
    """
    Perform the forward pass of the multiple kernel changepoint algorithm.
    :param X: Matrix of observations or pre-computed gram matrix
    :param min_cp: Lower bound on interval of the number of changepoints to detect
    :param max_cp: Upper bound on interval of the number of changepoints to detect
    :param kernel_type: Type of kernel to use:
                        -1: Precomputed gram stored in X
                        0: Chi-square 
                        1: Gaussian with Euclidean distance
                        2: Gaussian with Hellinger distance
                        3: Gaussian with TV distance
                        4: Linear 
    :param bw: Bandwidth for the kernel
    :param min_dist: Minimum allowable distance between successive changepoints
    :return: N: N[i,j] is argmin_{i<=s<=k} I[i-1,s] + \sum_{t=s+1}^k \|\phi(x_t)-\hat\mu_{s+1,k}\|^2_H.
             I: I[i,j] is the objective value for j changepoints in the first i observations
    """
    cdef double[::1] X_contig = X.flatten()
    cdef long n = np.size(X, 0)
    cdef unsigned int d = np.size(X, 1)
    cdef long area_table_size = n*(n+1)/2
    cdef double[:, ::1] I = np.inf*np.ones([n, max_cp], dtype=np.float64)
    cdef double[:, ::1] N = np.inf*np.ones([n, max_cp], dtype=np.int)
    cdef double* area_table_ptr = <double*> PyMem_Malloc(sizeof(double) * area_table_size)
    if not area_table_ptr:
        raise MemoryError()
    cdef double[::1] area_table = <double[:area_table_size]>area_table_ptr
    cdef double* diag_ptr = <double*> PyMem_Malloc(sizeof(double) * n)
    cdef double[::1] diag = <double[:n]>diag_ptr
    cdef long i, j, t, idx, i_end, t_end
    cdef double s, temp
    cdef double opt_obj=0.0
    cdef int num_threads, thread_id

    if max_cp > n:
        print('Maximum number of expected segments cannot exceed the size of the data vector.')
        raise AssertionError
    if d != n and kernel_type == -1:
        print('Expected pre-computed gram matrix since kernel_type=-1, but the input X is not square.')
        raise AssertionError

    num_threads = omp_get_max_threads()-1
    compute_area_table(area_table, diag, X_contig, n, d, kernel_type, bw)

    # No changepoint case
    for i in range(0, n-min_cp*min_dist+1):
        # All but linear, precomputed kernels have 1 on the diagonal of the gram matrix
        if kernel_type != 4 and kernel_type != -1:
            I[i, 0] = i+1 - 1.0/(i+1.0)*area_table[i*(i+1)/2+i]
        else:
            I[i, 0] = diag[i] - 1.0/(i+1.0)*area_table[i*(i+1)/2+i]

    with nogil:
        for j in range(1, max_cp):  # j: number of changepoints
            i_end = n-long_max(<long> (min_cp-j)*min_dist-1, <int> 0)
            for i in prange(j+1, i_end, schedule='dynamic', num_threads=num_threads):
                t_end = long_max(<long>j, <long>(i + 1 - min_dist))
                for t in range(j, t_end):  # t: possible location of next best changepoint
                    if i > t+1:
                        if kernel_type != 4 and kernel_type != -1:
                            s = i-t - 1.0/(i-t)*(area_table[i*(i+1)/2+i]+area_table[(t+1)*t/2+t]-2*area_table[(i+1)*i/2+t])
                        else:
                            s = diag[i] - diag[t] - 1.0/(i-t)*(area_table[i*(i+1)/2+i]+area_table[(t+1)*t/2+t]-2*area_table[(i+1)*i/2+t])
                    else:
                        s = 0
                    idx = <unsigned int>(j - 1)
                    temp = I[t, idx] + s
                    if I[i, j] > temp:
                        I[i, j] = temp
                        N[i, j] = t

    PyMem_Free(area_table_ptr)
    PyMem_Free(diag_ptr)
    return N, I


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef kcpe_backward_pass(double[:, ::1] N, double[:, ::1] I, int k, int n, int min_dist):
    """
    Compute the backward pass of the multiple kernel changepoint algorithm.
    :param N: N[i,j] is argmin_{i<=s<=k} I[i-1,s] + \sum_{t=s+1}^k \|\phi(x_t)-\hat\mu_{s+1,k}\|^2_H.
    :param I: I[i,j] is the objective value for j changepoints in the first i observations
    :param k: Number of changepoints
    :param n: Number of observations
    :param min_dist: Minimum allowable distance between changepoints
    :return: T: Indices of changepoints
             I[n - min_dist, k - 1]: Objective value at the optimum
    """
    cdef int[::1] T = np.zeros(k, dtype=np.int32)
    cdef unsigned int i

    T[k - 2] = <int> N[n - min_dist, k - 1]
    for i in range(k - 3, -1, -1):
        T[i] = <int> N[T[i + 1], i + 1]
    T[k - 1] = n-1

    return T, I[n - min_dist, k - 1]


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void compute_area_table(double[::1] S, double[::1] diag, double[::1] X, long n, unsigned int d, int kernel_type, float bw):
    """
    Compute the summed area table for the gram matrix. Since the kernels are symmetric, we can just compute the lower 
    (or upper) diagonal.     
    :param S: Pointer to summed area table
    :param diag: Pointer to diagonal of gram matrix
    :param X: Pointer to data matrix
    :param n: Number of samples
    :param d: Dimension of each sample
    :param kernel_type: Type of kernel to use:
                        -1: Precomputed gram stored in X
                        0: Chi-square 
                        1: Gaussian with Euclidean distance
                        2: Gaussian with Hellinger distance
                        3: Gaussian with TV distance
                        4: Linear 
    :param bw: Bandwidth for kernel
    """

    cdef int num_threads
    cdef long i, j, start, diag_size, diag_entry, idx1, idx2

    num_threads = omp_get_max_threads()-1

    if num_threads == 1:
        for i in range(n):
            for j in range(i+1):
                fill_area_table(S, diag, X, n, d, kernel_type, bw, i, j)
    else:
        with nogil:
            for start in range(2*n-1):
                idx1 = (start+1)/2
                idx2 = start/2
                diag_size = long_min(start/2+1, <int> n-idx1)
                for diag_entry in prange(0, diag_size, schedule='dynamic', num_threads=num_threads):
                    i = idx1+diag_entry
                    j = idx2-diag_entry
                    fill_area_table(S, diag, X, n, d, kernel_type, bw, i, j)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void fill_area_table(double[::1] S, double[::1] diag, double[::1] X, long n, unsigned int d, int kernel_type, float bw, long i, long j) nogil:
    """
    Compute and fill in entry (i,j) in the summed area table for the gram matrix.  
    (or upper) diagonal.     
    :param S: Pointer to summed area table
    :param diag: Pointer to diagonal of gram matrix
    :param X: Pointer to data matrix
    :param n: Number of samples
    :param d: Dimension of each sample
    :param kernel_type: Type of kernel to use:
                        -1: Precomputed gram stored in X
                        0: Chi-square 
                        1: Gaussian with Euclidean distance
                        2: Gaussian with Hellinger distance
                        3: Gaussian with TV distance
                        4: Linear 
    :param bw: Bandwidth for kernel
    :param i: row of entry to be computed
    :param j: column of entry to be computed
    """
    cdef long k
    cdef double entry

    k = (i+1)*i/2+j

    if kernel_type != -1:
        entry = compute_gram_entry(X, i, j, n, d, kernel_type, bw)
    else:
        entry = X[j+d*i]

    S[k] = entry
    if 0 < j < i:
        S[k] = S[k] - S[(i-1)*i/2+j-1] + S[(i-1)*i/2+j] + S[k-1]
    elif 0 < i == j:
        S[k] = S[k] - S[(i-1)*i/2+j-1] + S[j*(j+1)/2+i-1] + S[k-1]
    elif i > 0:
        S[k] = S[k] + S[(i-1)*i/2+j]
    elif j > 0:
        S[k] = S[k] + S[j*(j-1)/2+i]
    if i == j:
        if i != 0:
            diag[i] = diag[i-1] + entry
        else:
            diag[i] = entry



