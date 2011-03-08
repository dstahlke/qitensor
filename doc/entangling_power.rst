Entangling Power of Bipartite Unitary
=====================================

    >>> from qitensor import *
    >>> import numpy
    >>> import numpy.linalg
    >>> # operational spaces
    >>> hA = qubit('A')
    >>> hB = qubit('B')
    >>> # ancilla spaces
    >>> ha = qubit('a')
    >>> hb = qubit('b')

    >>> # create two maximally entangled states
    >>> state_A = (hA * ha).array(numpy.diag([1, 1]))
    >>> state_A
    HilbertArray(|A,a>,
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j]]))
    >>> state_A.normalize()
    >>> state_B = (hB * hb).array(numpy.diag([1, 1]))
    >>> state_B.normalize()
    >>> state = state_A * state_B
    >>> state.space
    |A,B,a,b>

    >>> phase = numpy.pi / 2
    >>> U = (hA * hB).O.eye()
    >>> U[1, 1, 1, 1] = numpy.exp(1j * phase)
    >>> U
    HilbertArray(|A,B><A,B|,
    array([[[[ 1.+0.j,  0.+0.j],
             [ 0.+0.j,  0.+0.j]],
    <BLANKLINE>
            [[ 0.+0.j,  1.+0.j],
             [ 0.+0.j,  0.+0.j]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[ 0.+0.j,  0.+0.j],
             [ 1.+0.j,  0.+0.j]],
    <BLANKLINE>
            [[ 0.+0.j,  0.+0.j],
             [ 0.+0.j,  0.+1.j]]]]))

    >>> state = U * state
    >>> cross = state.transpose(ha * hA)
    >>> cross.space
    |B,b><A,a|
    >>> (u, s, v) = cross.svd()
    >>> schmidt = numpy.diag(s.as_np_matrix())
    >>> abs(numpy.linalg.norm(schmidt) - 1) < 1e-14
    True
    >>> schmidt = numpy.real(schmidt * numpy.conj(schmidt))
    >>> entropy = sum([ -x*numpy.log(x)/numpy.log(2) for x in schmidt if x > 0 ])
    >>> "%.6g" % entropy
    '0.600876'
