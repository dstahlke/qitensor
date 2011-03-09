Sage: Hilbert Space Indexed by Group Elements
=============================================

    >>> from sage.all import * # doctest: +SKIP
    >>> import numpy # doctest: +SKIP
    >>> from qitensor import indexed_space # doctest: +SKIP
    >>> G = DihedralGroup(3) # doctest: +SKIP
    >>> ha = indexed_space('a', G.list(), dtype=SR) # doctest: +SKIP
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
    >>> # X_h := sum_g{ |hg><g| }
    >>> pauliX = lambda h: numpy.sum([ha.ket(h*g) * ha.bra(g) for g in G]) # doctest: +SKIP

    >>> pauliX(G[2]) # doctest: +SKIP
    |a><a|
    [0 0 1 0 0 0]
    [0 0 0 0 1 0]
    [1 0 0 0 0 0]
    [0 0 0 0 0 1]
    [0 1 0 0 0 0]
    [0 0 0 1 0 0]
    >>> pauliX(G[2]) * ha.ket(G[3]) == ha.ket(G[2] * G[3]) # doctest: +SKIP
    True
