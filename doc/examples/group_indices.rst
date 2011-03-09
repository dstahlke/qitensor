Sage: Hilbert Space Indexed by Group Elements
=============================================

    >>> from sage.all import *
    >>> import numpy
    >>> from qitensor import indexed_space
    >>> G = DihedralGroup(3)
    >>> ha = indexed_space('a', G.list(), dtype=SR)
    >>> ha
    |a>
    >>> ha.ket(G[1])
    |a>
    [0]
    [1]
    [0]
    [0]
    [0]
    [0]
    >>> ha.bra(G[1])
    <a|
    [0 1 0 0 0 0]

    >>> # Create group generalized Pauli X operator defined by
    >>> # X_h := sum_g{ |hg><g| }
    >>> pauliX = lambda h: numpy.sum([ha.ket(h*g) * ha.bra(g) for g in G])

    >>> pauliX(G[2])
    |a><a|
    [0 0 1 0 0 0]
    [0 0 0 0 1 0]
    [1 0 0 0 0 0]
    [0 0 0 0 0 1]
    [0 1 0 0 0 0]
    [0 0 0 1 0 0]
    >>> pauliX(G[2]) * ha.ket(G[3]) == ha.ket(G[2] * G[3])
    True
