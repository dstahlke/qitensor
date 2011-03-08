Sage: Hilbert Space Indexed by Group Elements
=============================================

::
    sage: import numpy
    sage: from qitensor import indexed_space
    sage: G = DihedralGroup(3)
    sage: ha = indexed_space('a', G.list(), dtype=SR)
    sage: ha
    |a>
    sage: ha.ket(G[1])
    |a>
    [0]
    [1]
    [0]
    [0]
    [0]
    [0]
    sage: ha.bra(G[1])
    <a|
    [0 1 0 0 0 0]

    sage: # Create group generalized Pauli X operator defined by
    sage: # X_h := sum_g{ |hg><g| }
    sage: pauliX = lambda h: numpy.sum([ha.ket(h*g) * ha.bra(g) for g in G])

    sage: pauliX(G[2])
    |a><a|
    [0 0 1 0 0 0]
    [0 0 0 0 1 0]
    [1 0 0 0 0 0]
    [0 0 0 0 0 1]
    [0 1 0 0 0 0]
    [0 0 0 1 0 0]
    sage: pauliX(G[2]) * ha.ket(G[3]) == ha.ket(G[2] * G[3])
    True
