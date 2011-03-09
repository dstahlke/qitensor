Sage: Hilbert Space Indexed by Group Elements
=============================================

    >>> from sage.all import * # doctest: +SKIP
    >>> import numpy # doctest: +SKIP
    >>> from qitensor import indexed_space # doctest: +SKIP
    >>> G = DihedralGroup(3) # doctest: +SKIP
    >>> ha = indexed_space('a', G.list(), dtype=SR) # doctest: +SKIP
    >>> hb = indexed_space('b', G.list(), dtype=SR) # doctest: +SKIP
    >>> ha # doctest: +SKIP
    |a>
    >>> ha.ket(G[1]) # doctest: +SKIP
    |a>
    [0]
    [1]
    [0]
    [0]
    [0]
    [0]
    >>> ha.bra(G[1]) # doctest: +SKIP
    <a|
    [0 1 0 0 0 0]

    >>> # Create group generalized Pauli X operator defined by
    >>> # X_h := sum_g{ |h*g><g| }
    >>> pauliX = lambda space, h: numpy.sum([space.ket(h*g) * space.bra(g) for g in G]) # doctest: +SKIP

    >>> pauliX(hb, G[2]) # doctest: +SKIP
    |b><b|
    [0 0 1 0 0 0]
    [0 0 0 0 1 0]
    [1 0 0 0 0 0]
    [0 0 0 0 0 1]
    [0 1 0 0 0 0]
    [0 0 0 1 0 0]
    >>> pauliX(ha, G[2]) * ha.ket(G[3]) == ha.ket(G[2] * G[3]) # doctest: +SKIP
    True

    >>> # Create controlled-group operator (an extension of CNOT).
    >>> # This maps |h,g> -> |h,h*g>
    >>> cmul = lambda sp1, sp2: numpy.sum([ sp1.ket(h) * sp1.bra(h) * pauliX(sp2, h) for h in G ]) # doctest: +SKIP

    >>> cmul(ha, hb) * ha.ket(G[2]) * hb.ket(G[3]) == ha.ket(G[2]) * hb.ket(G[2] * G[3]) # doctest: +SKIP
    True
