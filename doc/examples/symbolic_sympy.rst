Sympy Example
=============

    >>> import sympy
    >>> from qitensor import qubit

    >>> ha = qubit('a', dtype=sympy)
    >>> hb = qubit('b', dtype=sympy)
    >>> sympy.var('x y')
    (x, y)
    >>> s = ha.array([1, x])
    >>> t = hb.array([1, y])
    >>> s*t
    HilbertArray(|a,b>,
    array([[1, y],
           [x, x*y]], dtype=object))

    >>> # density operator
    >>> (s*t).O
    HilbertArray(|a,b><a,b|,
    array([[[[1, conjugate(y)],
             [conjugate(x), conjugate(x)*conjugate(y)]],
    <BLANKLINE>
            [[y, y*conjugate(y)],
             [y*conjugate(x), y*conjugate(x)*conjugate(y)]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[x, x*conjugate(y)],
             [x*conjugate(x), x*conjugate(x)*conjugate(y)]],
    <BLANKLINE>
            [[x*y, x*y*conjugate(y)],
             [x*y*conjugate(x), x*y*conjugate(x)*conjugate(y)]]]], dtype=object))
    >>> # density operator with ``ha`` traced out
    >>> (s*t).O.trace(ha)
    HilbertArray(|b><b|,
    array([[x*conjugate(x) + 1, x*conjugate(x)*conjugate(y) + conjugate(y)],
           [x*y*conjugate(x) + y,
            x*y*conjugate(x)*conjugate(y) + y*conjugate(y)]], dtype=object))

    >>> U = (ha * hb).eye()
    >>> # arrays can be indexed using dictionaries
    >>> U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 0 }] = x
    >>> U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 1 }] = y
    >>> U
    HilbertArray(|a,b><a,b|,
    array([[[[x, y],
             [0, 0]],
    <BLANKLINE>
            [[0, 1],
             [0, 0]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[0, 0],
             [1, 0]],
    <BLANKLINE>
            [[0, 0],
             [0, 1]]]], dtype=object))

    >>> U.I
    HilbertArray(|a,b><a,b|,
    array([[[[1/x, -y/x],
             [0, 0]],
    <BLANKLINE>
            [[0, 1],
             [0, 0]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[0, 0],
             [1, 0]],
    <BLANKLINE>
            [[0, 0],
             [0, 1]]]], dtype=object))

    >>> U * U.I
    HilbertArray(|a,b><a,b|,
    array([[[[1, 0],
             [0, 0]],
    <BLANKLINE>
            [[0, 1],
             [0, 0]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[0, 0],
             [1, 0]],
    <BLANKLINE>
            [[0, 0],
             [0, 1]]]], dtype=object))

    >>> ((U ** 3) * (U ** -3)).simplify_full()
    HilbertArray(|a,b><a,b|,
    array([[[[1, 0],
             [0, 0]],
    <BLANKLINE>
            [[0, 1],
             [0, 0]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[0, 0],
             [1, 0]],
    <BLANKLINE>
            [[0, 0],
             [0, 1]]]], dtype=object))
