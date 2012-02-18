Map-State Duality
=================

    >>> from qitensor import qubit, qudit
    >>> # input spaces for bipartite unitary
    >>> hA = qudit('A', 3)
    >>> hB = qudit('B', 4)
    >>> # output spaces for bipartite unitary
    >>> hAbar = qudit('Abar', 2)
    >>> hBbar = qudit('Bbar', 6)
    >>> # create a unitary
    >>> U = (hAbar*hBbar*hA.H*hB.H).random_unitary()
    >>> U.space
    |Abar,Bbar><A,B|

    >>> # the cross operator (partial transpose)
    >>> Ux = U.relabel({ hA.H: hA, hBbar: hBbar.H })
    >>> Ux.space
    |A,Abar><B,Bbar|
    >>> # do a Schmidt decomposition
    >>> hc = qudit('c', 6)
    >>> (VA, M, VB) = Ux.svd(inner_space=hc, full_matrices=False)
    >>> VA.space
    |A,Abar><c|
    >>> M.space
    |c><c|
    >>> VB.space
    |c><B,Bbar|
    >>> Ux.closeto(VA*M*VB)
    True

    >>> # VA is an isometry, so it is the channel ket of a CPTP map.
    >>> # Here is the superoperator for this map:
    >>> E = lambda x: (VA * x * VA.H).trace(hAbar)

    >>> # Similarly, construct a channel for HB, but this will act on space c'
    >>> VBprime = VB.relabel({ hc: hc.prime })
    >>> VBprime.space
    |c'><B,Bbar|
    >>> # How about transposing it...
    >>> VBprime = VBprime.T
    >>> VBprime.space
    |B,Bbar><c'|
    >>> # the superoperator:
    >>> F = lambda x: (VBprime * x * VBprime.H).trace(hBbar)

    >>> # The M matrix can be converted to a state by doing a partial
    >>> # transpose.  It is not allowed to have the same space repeated
    >>> # twice, so prime one of them.
    >>> psi = M.relabel({ hc.H: hc.prime }).normalized()
    >>> psi.space
    |c,c'>
    >>> # make a density operator
    >>> rho = psi.O # shortcut for writing psi*psi.H
    >>> rho.space
    |c,c'><c,c'|
    >>> # pass this through the channels
    >>> E(rho).space
    |A,c'><A,c'|
    >>> sigma = F(E(rho))
    >>> sigma.space
    |A,B><A,B|
    >>> # The result is the fully mixed state on |A,B>.  Can you see why?
    >>> sigma.closeto( (hA*hB).eye()/12 )
    True
