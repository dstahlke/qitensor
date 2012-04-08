cimport cpython

from qitensor.space cimport HilbertSpace

cpdef _cached_atom_factory(label, latex_label, indices, group_op, base_field)
cpdef _assert_all_compatible(collection)

cdef class HilbertAtom(HilbertSpace):
    cdef readonly str label
    cdef readonly str latex_label
    cdef readonly tuple indices
    cdef readonly group_op
    cdef readonly cpython.bool is_dual
    cdef readonly tuple key
    cdef readonly long _hashval
    cdef readonly HilbertAtom _prime

    cpdef _mycmp(self, other)
    cpdef _assert_compatible(self, other)
    cpdef ket(self, idx)
    cpdef bra(self, idx)
    cpdef fourier_basis_state(self, k)
    cpdef x_plus(self)
    cpdef x_minus(self)
    cpdef y_plus(self)
    cpdef y_minus(self)
    cpdef z_plus(self)
    cpdef z_minus(self)
    cpdef bloch(self, theta, phi)
    cpdef pauliX(self, h=*, left=*)
    cpdef pauliY(self)
    cpdef pauliZ(self, order=*)
    cpdef hadamard(self)
    cpdef gateS(self)
    cpdef gateT(self)
    cpdef _create_addend_isoms(self)
