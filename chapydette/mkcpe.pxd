import numpy as np
cimport numpy as np
cpdef kcpe(np.ndarray[np.float64_t, ndim=2] X, int k, int kernel_type, double bw, unsigned int min_dist)
cpdef kcpe_range_cp(np.ndarray[np.float64_t, ndim=2] X, int min_cp, int max_cp, int kernel_type, double bw, unsigned int min_dist)