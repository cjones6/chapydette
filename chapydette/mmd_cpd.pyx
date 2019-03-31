# coding=utf-8
from __future__ import division
import numpy as np


cpdef mmd_cpd(gram_matrix):
    """
    Given a gram matrix, estimate the location of one changepoint based on the Maximum Mean Discrepancy.

    References:
    - Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. Journal
     of Machine Learning Research, 13(Mar), 723-773.
    - Harchaoui, Z., Moulines, E., & Bach, F. R. (2009). Kernel change-point analysis. In Advances in Neural Information
     Processing Systems (pp. 609-616).
     
    :param X: Either the data matrix (if gram=False) or the gram matrix (if gram=True)
    :return t_opt: Estimated changepoint index. This is the last index in the first segment.
    :return obj_vals: The objective value at each location.
    """

    T = np.size(gram_matrix, 0)
    obj_vals = np.zeros(T-1)
    for t in range(1, T):
        term1 = (T-t)/(t*T)*np.sum(gram_matrix[0:t, 0:t])
        term2 = -2/T*np.sum(gram_matrix[0:t, t:T])
        term3 = t/((T-t)*T)*np.sum(gram_matrix[t:, t:])
        obj_vals[t-1] = term1 + term2 + term3
    t_opt = np.argmax(obj_vals)
    return t_opt, obj_vals


