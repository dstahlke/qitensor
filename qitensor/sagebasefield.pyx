#!/usr/bin/python

import numpy as np
cimport numpy as np
cimport cpython

import sage.all
from qitensor.basefield import HilbertBaseField
from qitensor.basefield cimport HilbertBaseField
from qitensor.arrayformatter import FORMATTER

cdef dict _base_field_cache = {}

# Cython doesn't yet support lambda in cpdef funcs, so this helper function is
# declared here.
cpdef do_simplify_full(x):
    return x.simplify_full()

cpdef _factory(dtype):
    """Don't call this, use base_field_lookup instead."""

    if not isinstance(dtype, sage.all.CommutativeRing):
        return None

    if not _base_field_cache.has_key(dtype):
        _base_field_cache[dtype] = SageHilbertBaseField(dtype)
    return _base_field_cache[dtype]

def _unreduce_v1(sage_ring):
    return _factory(sage_ring)

cdef class SageHilbertBaseField(HilbertBaseField):
    def __init__(self, sage_ring):
        """Don't call this, use base_field_lookup instead."""
        unique_id = 'sage '+repr(sage_ring)
        HilbertBaseField.__init__(self, object, unique_id)
        self.sage_ring = sage_ring

    def __reduce__(self):
        return _unreduce_v1, (self.sage_ring, )

    cpdef np.ndarray sage_mat_xform(self, np.ndarray m, f):
        return self.matrix_sage_to_np(f(self.matrix_np_to_sage(m)))

    cpdef complex_unit(self):
        return self.sage_ring(sage.all.I)

    cpdef latex_formatter(self, data, dollar_if_tex):
        return FORMATTER.sage_scalar_latex_formatter(data, dollar_if_tex)

    cpdef input_cast_function(self):
        return self.sage_ring

    cpdef fractional_phase(self, int a, int b):
        return self.sage_ring(sage.all.exp(2 * sage.all.pi * sage.all.I * a / b))

    cpdef sqrt(self, x):
        return self.sage_ring(sage.all.sqrt(x))

    cpdef xlog2x(self, x):
        return self.sage_ring(0 if x<=0 else x*sage.all.log(x)/sage.all.log2)

    cpdef np.ndarray eye(self, long size):
        return np.array(sage.all.identity_matrix(self.sage_ring, size), dtype=self.dtype)

    cpdef np.ndarray mat_n(self, np.ndarray m, prec=None, digits=None):
        return self.matrix_sage_to_np(self.matrix_np_to_sage(m).n(prec=prec, digits=digits))
        #return self.sage_mat_xform(m, \
        #    lambda x: x.n(prec=prec, digits=digits))

    cpdef np.ndarray mat_simplify(self, np.ndarray m, full=False):
        if full:
            return np.vectorize(do_simplify_full, otypes=[self.dtype])(m)
            #return m.apply_map(lambda x: x.simplify_full())
        else:
            return self.matrix_sage_to_np(self.matrix_np_to_sage(m).simplify())
            #return m.apply_map(lambda x: x.simplify())

    cpdef np.ndarray mat_adjoint(self, np.ndarray m):
        return self.matrix_sage_to_np(self.matrix_np_to_sage(m).conjugate().transpose())
        #return self.sage_mat_xform(m, lambda x: x.conjugate().transpose())

    cpdef np.ndarray mat_inverse(self, np.ndarray m):
        return self.matrix_sage_to_np(self.matrix_np_to_sage(m).inverse())
        #return self.sage_mat_xform(m, lambda x: x.inverse())

    cpdef mat_det(self, np.ndarray m):
        return self.matrix_np_to_sage(m).det()

    cpdef mat_norm(self, np.ndarray arr):
        return self.sqrt(np.sum(arr * np.conj(arr)))

    cpdef np.ndarray mat_conj(self, np.ndarray m):
        return self.matrix_sage_to_np(self.matrix_np_to_sage(m).conjugate())
        #return self.sage_mat_xform(m, lambda x: x.conjugate())

    cpdef np.ndarray mat_pow(self, np.ndarray m, n):
        return self.matrix_sage_to_np(self.matrix_np_to_sage(m) ** n)
        #return self.sage_mat_xform(m, lambda x: x**n)

    cpdef mat_eig(self, np.ndarray m, cpython.bool hermit):
        (w, v) = self.matrix_np_to_sage(m).eigenmatrix_right()

        # convert result to numpy
        w = np.array(w.diagonal())
        v = np.array(v)

        # Sage doesn't normalize the columns for symbolic expressions, so do it
        # here.
        v = np.array([c / sage.all.sqrt(np.sum(c**2)) for c in v.T]).T

        return (w, v)

    cpdef mat_eigvals(self, np.ndarray m, cpython.bool hermit):
        w = self.matrix_np_to_sage(m).eigenvalues()
        # convert result to numpy
        return np.array(w)
