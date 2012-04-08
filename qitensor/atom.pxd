from qitensor.space cimport HilbertSpace

cdef class HilbertAtom(HilbertSpace):
    cdef readonly label
    cdef readonly latex_label
    cdef readonly indices
    cdef readonly group_op
    cdef readonly is_dual
    cdef readonly key
    cdef readonly _hashval
    cdef readonly _prime
