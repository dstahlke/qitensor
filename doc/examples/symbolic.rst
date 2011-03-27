Sage: Vectors Over the Symbolic Ring
====================================

    >>> from sage.all import * # doctest: +SKIP
    >>> from qitensor import qubit # doctest: +SKIP
    >>> ha = qubit('a', dtype=SR) # doctest: +SKIP
    >>> hb = qubit('b', dtype=SR) # doctest: +SKIP
    >>> (x, y) = var('x y') # doctest: +SKIP
    >>> s = ha.array([1, x]) # doctest: +SKIP
    >>> t = hb.array([1, y]) # doctest: +SKIP
    >>> s*t # doctest: +SKIP
    |a,b>
    [  1]
    [  y]
    [---]
    [  x]
    [x*y]

    >>> U = (ha * hb).eye() # doctest: +SKIP
    >>> U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 0 }] = x # doctest: +SKIP
    >>> U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 1 }] = y # doctest: +SKIP
    >>> U # doctest: +SKIP
    |a,b><a,b|
    [x y|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1]

    >>> U.I # doctest: +SKIP
    |a,b><a,b|
    [ 1/x -y/x|   0    0]
    [   0    1|   0    0]
    [---------+---------]
    [   0    0|   1    0]
    [   0    0|   0    1]

    >>> U * U.I # doctest: +SKIP
    |a,b><a,b|
    [1 0|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1]

    >>> ((U ** 3) * (U ** -3)).apply_map(lambda x: x.simplify_full()) # doctest: +SKIP
    |a,b><a,b|
    [1 0|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1]
