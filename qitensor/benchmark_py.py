"""
A series of performance tests.
"""

import numpy as np

from qitensor import *

def random_channels(D=2, cnt=3000):
    ha = qudit('a', D)
    hb = qudit('b', D)
    hc = qudit('c', D*D)

    accum = hb.O.array()

    for idx in range(cnt):
        K = (hb*hc*ha.H).random_isometry()
        U = ha.random_unitary()
        rho = U * ha.diag(np.random.rand(D)) * U.H
        sigma = (K * rho * K.H).trace(hc)
        accum += sigma

    return accum / cnt

def orbit(D=2, cnt=50000):
    ha = qudit('a', D)
    hb = qudit('b', D)

    x = (ha*hb).random_unitary()
    y = (ha*hb).random_unitary()

    for idx in range(cnt):
        y *= x

    return y
