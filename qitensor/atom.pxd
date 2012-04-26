cimport cpython

from qitensor.space cimport HilbertSpace

cpdef _atom_factory(base_field, label, latex_label, indices, group_op)
cpdef _assert_all_compatible(collection)

cdef class HilbertAtom(HilbertSpace):
    cdef readonly str label
    cdef readonly str latex_label
    cdef readonly tuple indices
    cdef readonly group_op
    cdef readonly cpython.bool is_dual
    cdef readonly tuple key
    cdef readonly long _hashval

    cpdef _mycmp(self, other)
    cpdef _assert_compatible(self, HilbertAtom other)
    cpdef ket(self, idx)
    cpdef bra(self, idx)
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
