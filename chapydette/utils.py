from __future__ import division
from __future__ import print_function


def convert_cps_to_time(cps, interval_start_times, interval_end_times):
    """
    Convert the estimated change points to times.

    :param cps: Estimated change points (indices).
    :type cps: array-like
    :param interval_start_times: Start times of the sliding windows.
    :type interval_start_times: numpy.array
    :param interval_end_times: End times of the sliding windows.
    :type interval_end_times: numpy.array
    :return: cps_times: Estimated change points in terms of time.
    :rtype: list
    """
    cps_times = []
    for cp in cps:
        t1 = interval_end_times[cp]
        t2 = interval_start_times[cp+1]
        cps_times.append((t1+t2)/2)
    return cps_times
