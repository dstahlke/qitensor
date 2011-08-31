Sage: pickling
====================================

    >>> from sage.all import * # doctest: +SKIP
    >>> from qitensor import qubit # doctest: +SKIP
    >>> ha = qubit('a', dtype=SR) # doctest: +SKIP
    >>> hb = qubit('b', dtype=SR) # doctest: +SKIP
    >>> (x, y) = var('x y') # doctest: +SKIP
    >>> s = ha.array([1, x]) # doctest: +SKIP
    >>> t = hb.array([1, y]) # doctest: +SKIP
    >>> Q = s*t # doctest: +SKIP
    >>> Q
    |a,b>
    [  1]
    [  y]
    [---]
    [  x]
    [x*y]

    >>> R = loads(dumps(Q))
    >>> R
    |a,b>
    [  1]
    [  y]
    [---]
    [  x]
    [x*y]

    >>> Q == R
    True
    >>> Q is R
    False
    >>> Q.space is R.space
    True
