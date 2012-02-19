Sage: pickling
====================================

    >>> from sage.all import * # doctest: +SKIP
    >>> from qitensor import qubit, set_qitensor_printoptions # doctest: +SKIP
    >>> # use Sage's nice array formatting
    >>> set_qitensor_printoptions(str_use_sage=True) # doctest: +SKIP

    >>> ha = qubit('a', dtype=SR) # doctest: +SKIP
    >>> hb = qubit('b', dtype=SR) # doctest: +SKIP
    >>> (x, y) = var('x y') # doctest: +SKIP
    >>> s = ha.array([1, x]) # doctest: +SKIP
    >>> t = hb.array([1, y]) # doctest: +SKIP
    >>> Q = s*t # doctest: +SKIP
    >>> Q # doctest: +SKIP
    HilbertArray(|a,b>,
    [  1]
    [  y]
    [---]
    [  x]
    [x*y])

    >>> R = loads(dumps(Q)) # doctest: +SKIP
    >>> R # doctest: +SKIP
    HilbertArray(|a,b>,
    [  1]
    [  y]
    [---]
    [  x]
    [x*y])

    >>> Q == R # doctest: +SKIP
    True
    >>> Q is R # doctest: +SKIP
    False
    >>> Q.space is R.space # doctest: +SKIP
    True
