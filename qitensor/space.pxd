cimport cpython
cimport numpy as np

cpdef _cached_space_factory(ket_set, bra_set)
cpdef create_space1(kets_and_bras)
cpdef create_space2(frozenset ket_set, frozenset bra_set)

cdef class HilbertSpace(object):
    cdef HilbertSpace _H
    cdef readonly frozenset ket_set
    cdef readonly frozenset bra_set
    cdef readonly frozenset bra_ket_set
    cdef readonly list sorted_kets
    cdef readonly list sorted_bras
    cdef readonly tuple shape
    cdef readonly long _dim
    cdef readonly cpython.bool _is_simple_dyad
    cdef readonly list _array_axes
    cdef readonly dict _array_axes_lookup
    cdef readonly base_field

    # for direct sum
    cdef public addends
    cdef public P

    cpdef bra_space(self)
    cpdef ket_space(self)
    cpdef is_symmetric(self)
    cpdef is_square(self)
    cpdef assert_square(self)
    cpdef diag(self, v)
    cpdef reshaped_np_matrix(self, m, input_axes=*)
    cpdef array(self, data=*, cpython.bool noinit_data=*, cpython.bool reshape=*, input_axes=*)
    cpdef random_array(self)
    cpdef random_unitary(self)
    cpdef random_isometry(self)
    cpdef eye(self)
    cpdef basis_vec(self, idx)
    cpdef basis(self)
    cpdef hermitian_basis(self, normalize=*)
    cpdef full_space(self)
    cpdef empty_space(self)
    cpdef dim(self)
    cpdef index_iter(self)
    cpdef assert_ket_space(self)
    cpdef reshaped_sage_matrix(self, m, input_axes=*)
