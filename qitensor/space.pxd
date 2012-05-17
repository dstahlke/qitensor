cimport cpython
cimport numpy as np

from qitensor.array cimport HilbertArray
from qitensor.basefield cimport HilbertBaseField

cpdef create_space1(kets_and_bras)
cpdef create_space2(frozenset ket_set, frozenset bra_set)
cpdef long _shape_product(l)

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
    cdef readonly HilbertBaseField base_field
    cdef readonly HilbertSpace _prime

    # for direct sum
    cdef public addends
    cdef public P

    cpdef HilbertSpace bra_space(self)
    cpdef HilbertSpace ket_space(self)
    cpdef is_symmetric(self)
    cpdef is_square(self)
    cpdef assert_square(self)
    cpdef HilbertArray diag(self, v)
    cpdef reshaped_np_matrix(self, m, input_axes=*)
    cpdef array(self, data=*, cpython.bool noinit_data=*, cpython.bool reshape=*, input_axes=*)
    cpdef HilbertArray random_array(self)
    cpdef HilbertArray random_unitary(self)
    cpdef HilbertArray random_isometry(self)
    cpdef HilbertArray random_density(self)
    cpdef HilbertArray eye(self)
    cpdef basis_vec(self, idx)
    cpdef basis(self)
    cpdef hermitian_basis(self, normalize=*)
    cpdef HilbertArray fourier_basis_state(self, int k)
    cpdef HilbertArray fourier(self)
    cpdef full_space(self)
    cpdef empty_space(self)
    cpdef int dim(self)
    cpdef index_iter(self)
    cpdef assert_ket_space(self)
    cpdef reshaped_sage_matrix(self, m, input_axes=*)
