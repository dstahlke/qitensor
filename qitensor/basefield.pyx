"""
HilbertBaseField defines how the mathematics of HilbertArray works.  Normally
you don't need to worry about this because usually the default implementation
is appropriate.  It is not recommened to call the constructor directly.
Instead, use the interface provided by
:func:`qitensor.factory.base_field_lookup`, or just use the ``dtype``
parameter of the factory functions in :mod:`qitensor.factory`.  A subclass,
:class:`SageHilbertBaseField`, provides the ability to create arrays over Sage
types (e.g. SR).  This, too, is accessed through ``base_field_lookup`` or by
passing the ``dtype`` parameter.
"""

from __future__ import division

import numpy as np
cimport numpy as np
import numpy.random
import numpy.linalg

from qitensor.exceptions import HilbertError, MismatchedSpaceError
from qitensor.array import HilbertArray
from qitensor.arrayformatter import FORMATTER

__all__ = ['HilbertBaseField']

cdef dict _base_field_cache = {}

cpdef _factory(dtype):
    """Don't call this, use ``base_field_lookup`` instead."""

    if not isinstance(dtype, type):
        return None

    if not _base_field_cache.has_key(dtype):
        _base_field_cache[dtype] = HilbertBaseField(dtype, repr(dtype))
    return _base_field_cache[dtype]

def _unreduce_v1(dtype):
    """
    Handles restoring from pickle.
    """

    return _factory(dtype)

cdef class HilbertBaseField:
    def __init__(self, dtype, unique_id):
        """Don't call this, use base_field_lookup instead."""

        self.dtype = dtype
        self.unique_id = unique_id
        self.sage_ring = None

    def __reduce__(self):
        """
        Tells pickle how to store this object.
        """

        return _unreduce_v1, (self.dtype, )

    cpdef assert_same(self, HilbertBaseField other):
        """
        Asserts that this object is the same as other.
        """

        if self.unique_id != other.unique_id:
            raise MismatchedSpaceError('Different base_fields: '+
                repr(self.unique_id)+' vs. '+repr(other.unique_id))

    cpdef matrix_np_to_sage(self, np.ndarray np_mat, R=None):
        np_mat = np.array(np_mat)

        import sage.all
        if self.sage_ring is None:
            sage_mat = sage.all.matrix(np_mat)
        else:
            sage_mat = sage.all.matrix(self.sage_ring, np_mat)

        if R is None:
            return sage_mat
        else:
            return sage_mat.change_ring(R)

    cpdef np.ndarray matrix_sage_to_np(self, sage_mat):
        if self.sage_ring is not None:
            if sage_mat.base_ring() != self.sage_ring:
                sage_mat = sage_mat.change_ring(self.sage_ring)
        np_mat = np.matrix(sage_mat, dtype=self.dtype)
        return np_mat

    cpdef latex_formatter(self, data, dollar_if_tex):
        return FORMATTER.py_scalar_latex_formatter(data, dollar_if_tex)

    cpdef input_cast_function(self):
        return None

    cpdef complex_unit(self):
        return 1j

    cpdef infty(self):
        return np.infty

    cpdef fractional_phase(self, int a, int b):
        return np.exp(2j * np.pi * a / b)

    cpdef frac(self, p, q):
        return p/q

    cpdef sqrt(self, x):
        return np.sqrt(x)

    cpdef log2(self, x):
        return np.log2(x)

    cpdef xlog2x(self, x):
        return 0 if x<=0 else x*np.log2(x)

    cpdef np.ndarray random_array(self, shape):
        """Returns random array with standard normal distribution"""
        return (
            np.random.standard_normal(size=shape) +
            np.random.standard_normal(size=shape)*1j
        ) / np.sqrt(2)

    cpdef np.ndarray eye(self, long size):
        return np.eye(size)

    cpdef np.ndarray mat_adjoint(self, np.ndarray mat):
        return mat.H

    cpdef np.ndarray mat_inverse(self, np.ndarray mat):
        # linalg.inv is used instead of mat.I because the latter automatically
        # does pinv for non-square matrices, which is not really an inverse.
        # If you need pinv, just call the pinv method.
        return np.linalg.inv(mat)

    cpdef mat_det(self, np.ndarray mat):
        return np.linalg.det(mat)

    cpdef mat_norm(self, np.ndarray arr, p):
        if p == 2:
            return np.linalg.norm(arr)
        elif np.isposinf(p):
            return np.max(np.abs(arr))
        else:
            return np.sum(np.abs(arr)**p)**(1.0/p)

    cpdef np.ndarray mat_pinv(self, np.ndarray mat, rcond):
        return np.linalg.pinv(mat, rcond)

    cpdef np.ndarray mat_conj(self, np.ndarray mat):
        return mat.conj()

    cpdef np.ndarray mat_n(self, np.ndarray mat, prec=None, digits=None): # pylint: disable=W0613
        # arrays in this base field are already numeric
        return mat

    cpdef np.ndarray mat_simplify(self, np.ndarray mat, full=False): # pylint: disable=W0613
        return mat

    cpdef np.ndarray mat_expm(self, np.ndarray mat):
        import scipy.linalg
        return scipy.linalg.expm(mat)

    cpdef np.ndarray mat_logm(self, np.ndarray mat):
        import scipy.linalg
        return scipy.linalg.logm(mat, True)

    cpdef np.ndarray mat_pow(self, np.ndarray mat, n):
        return mat**n

    cpdef mat_svd(self, np.ndarray mat, full_matrices):
        # cast to complex in case we have symbolic vals from Sage
        (u, s, v) = np.linalg.svd(np.matrix(mat, dtype=complex), \
            full_matrices=full_matrices)
        return (u, s, v)

    cpdef mat_svd_vals(self, np.ndarray mat):
        # cast to complex in case we have symbolic vals from Sage
        (_u, s, _v) = np.linalg.svd(np.matrix(mat, dtype=complex), \
            full_matrices=False)
        return s

    cpdef mat_eig(self, np.ndarray mat, cpython.bool hermit):
        eig_fn = np.linalg.eigh if hermit else np.linalg.eig
        # cast to complex in case we have symbolic vals from Sage
        (w, v) = eig_fn(np.matrix(mat, dtype=complex))
        return (w, v)

    cpdef mat_eigvals(self, np.ndarray mat, cpython.bool hermit):
        eig_fn = np.linalg.eigvalsh if hermit else np.linalg.eigvals
        # cast to complex in case we have symbolic vals from Sage
        w = eig_fn(np.matrix(mat, dtype=complex))
        return w

    cpdef mat_qr(self, np.ndarray mat):
        # cast to complex in case we have symbolic vals from Sage
        (q, r) = np.linalg.qr(np.matrix(mat, dtype=complex))
        return (q, r)
