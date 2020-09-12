import numpy as np
import scipy
import scipy.special


def ach_slope_heuristic(objs, n, min_ncp, max_ncp, alpha=2):
    """
    Estimate the number of change points in the sequence using the slope heuristic with the penalty of Arlot et al.
    (2019).

    Reference:
    S. Arlot, A. Celisse, and Z. Harchaoui. A kernel multiple change-point algorithm via model selection. Journal of Machine Learning Research, 20(162):1–56, 2019.

    :param objs: Objective values for varying numbers of change points (at least dmin through dmax)
    :param n: Length of the sequence
    :param min_ncp: Minimum number of change points in the sequence
    :param max_ncp: Maximum number of change points in the sequence
    :param alpha: Parameter in the slope heuristic
    :return: est_ncp: Estimated number of change points in the sequence
    """
    # Regress the objective values on log (n-1 choose ncp) and ncp+1 and store the resultant slopes
    ncp_range_subset = np.array(range(max(int(0.6*(max_ncp)), min_ncp), max_ncp))
    x1_subset = scipy.special.loggamma(n*np.ones_like(ncp_range_subset)) - scipy.special.loggamma(n - ncp_range_subset)\
                - scipy.special.loggamma(ncp_range_subset + 1)
    x2_subset = ncp_range_subset+1
    x = np.column_stack((np.ones_like(x1_subset), x1_subset, x2_subset))
    objs_subset = np.array([objs[ncp] for ncp in ncp_range_subset])
    beta = np.linalg.solve(x.T.dot(x), x.T.dot(objs_subset))
    s1, s2 = beta[1], beta[2]

    # Obtain the penalized objectives
    ncp_range = np.array(range(min_ncp, max_ncp+1))
    x1 = scipy.special.loggamma(n*np.ones_like(ncp_range)) - scipy.special.loggamma(n - ncp_range) \
                - scipy.special.loggamma(ncp_range + 1)
    x2 = ncp_range+1
    objs_all = np.array([objs[key] for key in ncp_range]).flatten()
    penalty = -alpha*(s1*np.array(x1) + s2*np.array(x2))
    penalized_objs = objs_all + penalty

    # Find the estimated number of change points
    est_ncp = np.argmin(penalized_objs) + min_ncp
    penalized_objs = {cp: penalized_objs[i] for i, cp in enumerate(ncp_range)}

    return est_ncp, penalized_objs
