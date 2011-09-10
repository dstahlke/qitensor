"""
This module contains functions related to quantum circuits.
"""

from qitensor import HilbertAtom, HilbertError

__all__ = ['cphase', 'cnot', 'max_entangled']

def cphase(h1, h2):
    """
    Returns the controlled-phase or generalized controlled-phase gate.

    The given spaces must be HilbertAtom spaces (i.e. not tensor products).

    For qubits, returns an operator in h1*h2 given by the diagonal
    array [1, 1, 1, -1].

    For larger dimension spaces, h1 and h2 must be of the same dimension (call
    it `D`) and the operator returned is given by
    :math:`\sum_{j,k} e^{2 \pi i j k / D} |j,k><j,k|`,
    where `j` and `k` are taken to be integer indices, regardless of the actual
    contents of the space's index set.

    >>> from qitensor import qubit, cphase
    >>> ha = qubit('a')
    >>> hb = qubit('b')
    >>> cphase(ha, hb).as_np_matrix()
    matrix([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]])
    """

    for h in (h1, h2):
        if not isinstance(h, HilbertAtom):
            raise TypeError('spaces must be instance of HilbertAtom')

    field = h1.base_field

    D1 = len(h1.indices)
    D2 = len(h2.indices)
    if D1 != D2:
        raise HilbertError('spaces must be of the same dimension')

    ret = (h1*h2).O.array()
    for j in range(len(h1.indices)):
        for k in range(len(h2.indices)):
            ret[j, k, j, k] = field.fractional_phase(j*k, D1)

    return ret

def cnot(h1, h2):
    """
    Returns the controlled-not (controlled-X) gate.

    The given spaces must be HilbertAtom spaces (i.e. not tensor products), and
    must be qubits.

    >>> from qitensor import qubit, cnot
    >>> ha = qubit('a')
    >>> hb = qubit('b')
    >>> cnot(ha, hb).as_np_matrix()
    matrix([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
            [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j]])
    >>> cnot(hb, ha).as_np_matrix()
    matrix([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
            [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
            [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])
    """

    for h in (h1, h2):
        if not isinstance(h, HilbertAtom):
            raise TypeError('spaces must be instance of HilbertAtom')

    if len(h1.indices) != 2 or len(h2.indices) != 2:
        raise NotImplementedError("cnot is only implemented for qubits")

    ret = (h1*h2).eye()
    ret[{ h1: h1.indices[1], h1.H: h1.indices[1] }] = [[0, 1], [1, 0]]

    return ret

def max_entangled(h1, h2):
    """
    Returns a maximally entangled state.

    The given spaces must be HilbertAtom spaces (i.e. not tensor products).
    The spaces must also be of equal dimension (call it `D`).  The returned state is
    :math:`\sum_j (1/\sqrt{D}) |j,j><j,j|`, where `j` is taken to be an integer
    index, regardless of the actual contents of the space's index set.

    >>> from qitensor import qubit, qudit, indexed_space, max_entangled

    >>> ha = qubit('a')
    >>> hb = qubit('b')
    >>> max_entangled(ha, hb)
    HilbertArray(|a,b>,
    array([[ 0.707107+0.j,  0.000000+0.j],
           [ 0.000000+0.j,  0.707107+0.j]]))

    >>> ha = qudit('a', 4)
    >>> hb = qudit('b', 4)
    >>> max_entangled(ha, hb)
    HilbertArray(|a,b>,
    array([[ 0.5+0.j,  0.0+0.j,  0.0+0.j,  0.0+0.j],
           [ 0.0+0.j,  0.5+0.j,  0.0+0.j,  0.0+0.j],
           [ 0.0+0.j,  0.0+0.j,  0.5+0.j,  0.0+0.j],
           [ 0.0+0.j,  0.0+0.j,  0.0+0.j,  0.5+0.j]]))

    >>> ha = qudit('a', 4)
    >>> hb = indexed_space('b', ['w', 'x', 'y', 'z'])
    >>> max_entangled(ha, hb)
    HilbertArray(|a,b>,
    array([[ 0.5+0.j,  0.0+0.j,  0.0+0.j,  0.0+0.j],
           [ 0.0+0.j,  0.5+0.j,  0.0+0.j,  0.0+0.j],
           [ 0.0+0.j,  0.0+0.j,  0.5+0.j,  0.0+0.j],
           [ 0.0+0.j,  0.0+0.j,  0.0+0.j,  0.5+0.j]]))
    """

    for h in (h1, h2):
        if not isinstance(h, HilbertAtom):
            raise TypeError('spaces must be instance of HilbertAtom')

    field = h1.base_field

    D1 = len(h1.indices)
    D2 = len(h2.indices)
    if D1 != D2:
        raise HilbertError('spaces must be of the same dimension')

    return (h1*h2).array(field.eye(D1) / field.sqrt(D1), reshape=True)
