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

import numpy as np
import numpy.random
import numpy.linalg

from qitensor.exceptions import HilbertError, IncompatibleBaseFieldError
import qitensor.atom
import qitensor.space
from qitensor.array import HilbertArray

__all__ = ['HilbertBaseField']

_base_field_cache = {}

def _factory(dtype):
    """Don't call this, use ``base_field_lookup`` instead."""

    if not isinstance(dtype, type):
        return None

    if not _base_field_cache.has_key(dtype):
        _base_field_cache[dtype] = HilbertBaseField(dtype, repr(dtype))
    return _base_field_cache[dtype]

def _unreduce_v1(dtype):
    return _factory(dtype)

class HilbertBaseField(object):
    def __init__(self, dtype, unique_id):
        """Don't call this, use base_field_lookup instead."""

        self.dtype = dtype
        self.unique_id = unique_id
        self.sage_ring = None

    def __reduce__(self):
        return _unreduce_v1, (self.dtype, )

    def assert_same(self, other):
        if self.unique_id != other.unique_id:
            raise IncompatibleBaseFieldError('Different base_fields: '+
                repr(self.unique_id)+' vs. '+repr(other.unique_id))

    def matrix_np_to_sage(self, np_mat, R=None):
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

    def matrix_sage_to_np(self, sage_mat):
        if self.sage_ring is not None:
            if sage_mat.base_ring() != self.sage_ring:
                sage_mat = sage_mat.change_ring(self.sage_ring)
        np_mat = np.matrix(sage_mat, dtype=self.dtype)
        return np_mat

    def latex_formatter(self, data):
        if self.dtype == complex:
            # FIXME - print options should be configurable (take from numpy options?)
            return np.core.arrayprint.ComplexFormat(data, 6, True)
        else:
            return str

    def input_cast_function(self):
        return None

    def complex_unit(self):
        return 1j

    def fractional_phase(self, a, b):
        return np.exp(2j * np.pi * a / b)

    def sqrt(self, x):
        return np.sqrt(x)

    def xlog2x(self, x):
        return 0 if x<=0 else x*np.log2(x)

    def random_array(self, shape):
        real = numpy.random.random((shape)) * 2.0 - 1.0
        imag = numpy.random.random((shape)) * 2.0 - 1.0
        return real + 1j*imag

    def eye(self, size):
        return np.eye(size)

    def mat_adjoint(self, mat):
        return mat.H

    def mat_inverse(self, mat):
        # linalg.inv is used instead of mat.I because the latter automatically
        # does pinv for non-square matrices, which is not really an inverse.
        # If you need pinv, just call the pinv method.
        return np.linalg.inv(mat)

    def mat_det(self, mat):
        return np.linalg.det(mat)

    def mat_norm(self, arr):
        return np.linalg.norm(arr)

    def mat_pinv(self, mat, rcond):
        return np.linalg.pinv(mat, rcond)

    def mat_conj(self, mat):
        return mat.conj()

    def mat_n(self, mat, prec=None, digits=None): # pylint: disable=W0613
        # arrays in this base field are already numeric
        return mat

    def mat_simplify(self, mat, full=False): # pylint: disable=W0613
        return mat

    def mat_expm(self, mat, q):
        import scipy.linalg
        return scipy.linalg.expm(mat, q)

    def mat_pow(self, mat, n):
        return mat**n

    def mat_svd(self, mat, full_matrices):
        # cast to complex in case we have symbolic vals from Sage
        (u, s, v) = np.linalg.svd(np.matrix(mat, dtype=complex), \
            full_matrices=full_matrices)
        return (u, s, v)

    def mat_svd_vals(self, mat):
        # cast to complex in case we have symbolic vals from Sage
        (_u, s, _v) = np.linalg.svd(np.matrix(mat, dtype=complex), \
            full_matrices=False)
        return s

    def mat_eig(self, mat, hermit):
        eig_fn = np.linalg.eigh if hermit else np.linalg.eig
        # cast to complex in case we have symbolic vals from Sage
        (w, v) = eig_fn(np.matrix(mat, dtype=complex))
        return (w, v)

    def mat_eigvals(self, mat, hermit):
        eig_fn = np.linalg.eigvalsh if hermit else np.linalg.eigvals
        # cast to complex in case we have symbolic vals from Sage
        w = eig_fn(np.matrix(mat, dtype=complex))
        return w

    def mat_qr(self, mat):
        # cast to complex in case we have symbolic vals from Sage
        (q, r) = np.linalg.qr(np.matrix(mat, dtype=complex))
        return (q, r)

    def create_space1(self, kets_and_bras):
        r"""
        Creates a ``HilbertSpace`` from a collection of ``HilbertAtom`` objects.

        This provides an alternative to using the multiplication operator
        to combine ``HilbertAtom`` objects.

        :param kets_and_bras: a collection of ``HilbertAtom`` objects

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> field = ha.base_field
        >>> ha * hb == field.create_space1([ha, hb])
        True
        >>> ha.H * hb == field.create_space1([ha.H, hb])
        True
        """
        return self.create_space2(
            frozenset([x for x in kets_and_bras if not x.is_dual]),
            frozenset([x for x in kets_and_bras if x.is_dual]))

    def create_space2(self, ket_set, bra_set):
        r"""
        Creates a ``HilbertSpace`` from frozensets of ``HilbertAtom`` kets and bras.

        This provides an alternative to using the multiplication operator
        to combine ``HilbertAtom`` objects.

        :param ket_set: a collection of ``HilbertAtom`` objects for which ``is_dual==False``
        :param bra_set: a collection of ``HilbertAtom`` objects for which ``is_dual==True``

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> field = ha.base_field
        >>> ha * hb == field.create_space2(frozenset([ha, hb]), frozenset())
        True
        >>> ha.H * hb == field.create_space2(frozenset([hb]), frozenset([ha.H]))
        True
        """

        # Just return the atoms if possible:
        if len(ket_set) == 0 and len(bra_set) == 0:
            raise HilbertError('tried to create empty HilbertSpace')
        elif len(ket_set) == 1 and len(bra_set) == 0:
            return list(ket_set)[0]
        elif len(ket_set) == 0 and len(bra_set) == 1:
            return list(bra_set)[0]
        else:
            return self._space_factory(ket_set, bra_set)

    def _atom_factory(self, label, latex_label, indices, group_op):
        r"""
        Factory method for creating ``HilbertAtom`` objects.

        Subclasses can override this method in order to return custom
        subclasses of ``HilbertAtom``.

        Users should call methods in ``qitensor.factory`` instead.
        """
        return qitensor.atom._cached_atom_factory( \
            label, latex_label, indices, group_op, self)

    def _space_factory(self, ket_set, bra_set):
        r"""
        Factory method for creating ``HilbertSpace`` objects.

        Subclasses can override this method in order to return custom
        subclasses of ``HilbertSpace``.

        Users shouldn't call this function.
        """
        return qitensor.space._cached_space_factory(ket_set, bra_set)

    def _array_factory(self, space, data, noinit_data, reshape, input_axes):
        r"""
        Factory method for creating ``HilbertArray`` objects.

        Subclasses can override this method in order to return custom
        subclasses of ``HilbertArray``.

        Users shouldn't call this function.
        """
        return HilbertArray(space, data, noinit_data, reshape, input_axes)
