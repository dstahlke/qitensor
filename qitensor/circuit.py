"""
This module contains functions related to quantum circuits.
"""

from qitensor import HilbertAtom

__all__ = ['cnot']

def cnot(h1, h2):
    """
    Returns the C-not or generalized C-not gate.

    For qubits, returns an operator in h1*h2 given by the diagonal
    array [1, 1, 1, -1].

    For larger dimension spaces, h1 and h2 must be of the same dimension (call
    it `D`) and the operator returned is given by
    :math:`\sum_{j,k} e^{2 \pi i j k / D} |j,k><j,k|`.

    >>> from qitensor import qubit, cnot
    >>> ha = qubit('a')
    >>> hb = qubit('b')
    >>> cnot(ha, hb)
    HilbertArray(|a,b><a,b|,
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
             [ 0.+0.j, -1.+0.j]]]]))
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
