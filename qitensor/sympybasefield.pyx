#!/usr/bin/python

import numpy as np
cimport numpy as np
cimport cpython
import sympy

from qitensor.basefield import HilbertBaseField
from qitensor.basefield cimport HilbertBaseField
from qitensor.arrayformatter import FORMATTER

cdef HilbertBaseField _base_field_cache = SympyHilbertBaseField()

# Cython doesn't yet support lambda in cpdef funcs, so this helper function is
# declared here.
cpdef do_simplify_full(x):
    return x.simplify_full()

cpdef do_cast_to_sympy(x):
    # FIXME - best way?
    return sympy.Integer(0) + x

cpdef to_n(x):
    return sympy.N(x)

cpdef do_conj(x):
    return sympy.conjugate(x)

cpdef _factory(dtype):
    """Don't call this, use base_field_lookup instead."""

    if dtype == sympy:
        return _base_field_cache
    else:
        return None

def _unreduce_v1():
    return _factory(sympy)

cdef class SympyHilbertBaseField(HilbertBaseField):
    def __init__(self):
        """Don't call this, use base_field_lookup instead."""
        unique_id = 'sympy'
        HilbertBaseField.__init__(self, object, unique_id)

    def __reduce__(self):
        return _unreduce_v1, tuple()

    cpdef complex_unit(self):
        return sympy.I

    cpdef latex_formatter(self, data, dollar_if_tex):
        return FORMATTER.sympy_scalar_latex_formatter(data, dollar_if_tex)

    cpdef input_cast_function(self):
        return do_cast_to_sympy

    cpdef fractional_phase(self, int a, int b):
        return sympy.exp(2 * sympy.pi * sympy.I * sympy.Rational(a, b))

    cpdef np.ndarray eye(self, long size):
        return np.diag([sympy.Integer(1)] * size)

    cpdef sqrt(self, x):
        return sympy.sqrt(x)

    cpdef xlog2x(self, x):
        return 0 if x<=0 else x*sympy.log(x)/sympy.log(2)

    cpdef np.ndarray mat_n(self, np.ndarray m, prec=None, digits=None):
        # FIXME - handle digits param
        return np.vectorize(to_n, otypes=[self.dtype])(m)

    cpdef np.ndarray mat_simplify(self, np.ndarray m, full=False):
        return np.vectorize(sympy.simplify, otypes=[self.dtype])(m)

    cpdef np.ndarray mat_inverse(self, np.ndarray m):
        sm = sympy.Matrix(m)
        sm = sm.inverse_GE()
        return np.matrix(sm)

    cpdef mat_det(self, np.ndarray m):
        sm = sympy.Matrix(m)
        return sm.det()

    cpdef mat_norm(self, np.ndarray arr):
        return self.sqrt(np.sum(arr * np.conj(arr)))

    cpdef np.ndarray mat_conj(self, np.ndarray mat):
        return np.vectorize(do_conj, otypes=[self.dtype])(mat)

    cpdef np.ndarray mat_adjoint(self, np.ndarray mat):
        return np.transpose(self.mat_conj(mat))

    cpdef np.ndarray mat_pow(self, np.ndarray m, n):
        sm = sympy.Matrix(m)
        return np.matrix(sm ** n)
