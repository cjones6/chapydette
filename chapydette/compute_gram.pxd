import numpy as np
cimport numpy as np
cdef void compute_gram_c(double[::1] X, double *gram, unsigned int n, unsigned int d, int kernel_type, float bw)
cdef double compute_gram_entry(double[::1] X, long i, long j, long n, unsigned int d, int kernel_type, float bw) nogil
cpdef compute_gram(np.ndarray[np.float64_t, ndim=2] X, int kernel_type, float bw)