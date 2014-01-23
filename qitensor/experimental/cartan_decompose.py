import qitensor
from . import cartan_decompose_impl

def unitary_to_cartan(U):
    """
    Decomposes a bipartite qubit unitary into the form
    :math:`(U_A \\otimes U_B) e^{-i \\sum \\alpha_k \\sigma_{Ak} \\otimes \\sigma_{Bk}} (V_A \\otimes V_B)`
    where
    :math:`\\alpha_k \\in (-\\pi/4, \\pi/4]`.
    Five values are returned from this function: (UA, UB, VA, VB, alpha).

    >>> from qitensor import qubit
    >>> from qitensor.experimental.cartan_decompose import unitary_to_cartan
    >>> from qitensor.experimental.cartan_decompose import unitary_from_cartan
    >>> ha = qubit('a')
    >>> hb = qubit('b')
    >>> U = (ha*hb).O.random_unitary()
    >>> (UA, UB, VA, VB, alpha) = unitary_to_cartan(U)
    >>> Ud = unitary_from_cartan(ha*hb, alpha)
    >>> # experimental module doesn't always work, so skip doctest
    >>> (UA * UB * Ud * VA * VB - U).norm() < 1e-12 # doctest: +SKIP
    True
    """

    assert U.space == U.H.space
    assert len(U.space.ket_set) == 2
    (ha, hb) = U.space.ket_set
    assert len(ha.indices) == 2
    assert len(hb.indices) == 2

    (UA, UB, VA, VB, alpha) = cartan_decompose_impl.unitary_to_cartan( \
        U.as_np_matrix())

    UA = ha.O.array(UA)
    UB = hb.O.array(UB)
    VA = ha.O.array(VA)
    VB = hb.O.array(VB)

    return (UA, UB, VA, VB, alpha)

def unitary_from_cartan(space, alpha):
    """
    Returns :math:`e^{-i \\sum \\alpha_k \\sigma_{Ak} \\otimes \\sigma_{Bk}}`.
    The ``space`` parameter should be either a bipartite qubit ket space or
    operator space.
    """
    if len(space.bra_set) == 0:
        space = space.O
    U = cartan_decompose_impl.unitary_from_cartan(alpha)
    return space.reshaped_np_matrix(U)

#def test():
#    ha = qitensor.qubit('a')
#    hb = qitensor.qubit('b')
#    U = (ha*hb).O.random_unitary()
#    (UA, UB, VA, VB, alpha) = unitary_to_cartan(U)
#    Ud = unitary_from_cartan(ha*hb, alpha)
#    print (UA * UB * Ud * VA * VB - U).norm()
