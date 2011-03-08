Symbolics in Sage
=====================================

::
    sage: from qitensor import qubit
    sage: ha = qubit('a', SR)
    sage: hb = qubit('b', SR)
    sage: (x, y) = var('x y')
    sage: s = ha.array([1, x])
    sage: t = hb.array([1, y])
    sage: s*t
    |a,b>
    [  1]
    [  y]
    [---]
    [  x]
    [x*y]

    sage: U = (ha * hb).eye()
    sage: U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 0 }] = x
    sage: U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 1 }] = y
    sage: U
    |a,b><a,b|
    [x y|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1]

    sage: U.I
    |a,b><a,b|
    [ 1/x -y/x|   0    0]
    [   0    1|   0    0]
    [---------+---------]
    [   0    0|   1    0]
    [   0    0|   0    1]

    sage: U * U.I
    |a,b><a,b|
    [1 0|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1]
