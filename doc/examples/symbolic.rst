Sage: Vectors Over the Symbolic Ring
====================================

    >>> from sage.all import *
    >>> from qitensor import qubit
    >>> ha = qubit('a', dtype=SR)
    >>> hb = qubit('b', dtype=SR)
    >>> (x, y) = var('x y')
    >>> s = ha.array([1, x])
    >>> t = hb.array([1, y])
    >>> s*t
    |a,b>
    [  1]
    [  y]
    [---]
    [  x]
    [x*y]

    >>> U = (ha * hb).eye()
    >>> U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 0 }] = x
    >>> U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 1 }] = y
    >>> U
    |a,b><a,b|
    [x y|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1]

    >>> U.I
    |a,b><a,b|
    [ 1/x -y/x|   0    0]
    [   0    1|   0    0]
    [---------+---------]
    [   0    0|   1    0]
    [   0    0|   0    1]

    >>> U * U.I
    |a,b><a,b|
    [1 0|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1]
