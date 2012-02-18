Entangling Power of Bipartite Unitary
=====================================

This is not the simplest way to measure the entanglement of a bipartite unitary, but it makes for a good example.
Alice and Bob each have a maximally entangled pair of qubits.  They each pass one end of one of the entangled pairs
into their part of the unitary.  At the end, the resulting entanglement is computed.

    >>> from qitensor import qubit, max_entangled
    >>> import numpy as np
    >>> # operational spaces
    >>> hA = qubit('A')
    >>> hB = qubit('B')
    >>> # ancilla spaces
    >>> ha = qubit('a')
    >>> hb = qubit('b')

    >>> # create two maximally entangled states
    >>> state_A = max_entangled(hA, ha)
    >>> state_A
    HilbertArray(|A,a>,
    array([[ 0.707107+0.j,  0.000000+0.j],
           [ 0.000000+0.j,  0.707107+0.j]]))
    >>> state_B = max_entangled(hB, hb)
    >>> state = state_A * state_B
    >>> state.space
    |A,B,a,b>

    >>> # create controlled-phase gate
    >>> phase = np.pi / 2
    >>> U = (hA * hB).O.eye()
    >>> U[1, 1, 1, 1] = np.exp(1j * phase)
    >>> U.as_np_matrix()
    matrix([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j]])

    >>> # pass the entangled state into the unitary and measure the
    >>> # entanglement of the result
    >>> state = U * state
    >>> state.space
    |A,B,a,b>
    >>> # the easy way to find entanglement across a*A-b*B cut
    >>> "%.6g" % state.O.trace(ha*hA).entropy()
    '0.600876'
    >>> # the hard way to find entanglement across a*A-b*B cut
    >>> (u, schmidt, v) = state.svd_list(row_space=hA*ha)
    >>> schmidt = np.abs(schmidt * schmidt)
    >>> abs(np.sum(schmidt) - 1) < 1e-14
    True
    >>> entropy = sum([ -x*np.log(x)/np.log(2) for x in schmidt if x > 0 ])
    >>> "%.6g" % entropy
    '0.600876'
