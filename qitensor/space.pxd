cdef class HilbertSpace(object):
    cdef _H
    cdef readonly ket_set
    cdef readonly bra_set
    cdef readonly bra_ket_set
    cdef readonly sorted_kets
    cdef readonly sorted_bras
    cdef readonly shape
    cdef readonly _dim
    cdef readonly _is_simple_dyad
    cdef readonly _array_axes
    cdef readonly _array_axes_lookup
    cdef readonly base_field

    # for direct sum
    cdef public addends
    cdef public P
