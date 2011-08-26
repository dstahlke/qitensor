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

from qitensor import shape_product
from qitensor.exceptions import *
from qitensor.atom import HilbertAtom
from qitensor.array import HilbertArray
from qitensor.space import HilbertSpace

__all__ = ['HilbertBaseField']

class GroupOpCyclic(object):
    def __init__(self, D):
        self.D = D

    def op(self, x, y):
        return (x+y) % self.D

class GroupOpTimes(object):
    def __init__(self):
        pass

    def op(self, x, y):
        return x*y

class HilbertBaseField(object):
    def __init__(self, dtype, unique_id):
        self.dtype = dtype
        self.unique_id = unique_id
        self.sage_ring = None

    def assert_same(self, other):
        if self.unique_id != other.unique_id:
            raise IncompatibleBaseFieldError('Different base_fields: '+
                repr(self.unique_id)+' vs. '+repr(other.unique_id))

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

    def mat_adjoint(self, m):
        return m.np_matrix_transform(
            lambda x: x.H, transpose_dims=True)

    def mat_inverse(self, m):
        # linalg.inv is used instead of m.I because the latter automatically
        # does pinv for non-square matrices, which is not really an inverse.
        # If you need pinv, just call the pinv method.
        return m.np_matrix_transform(np.linalg.inv, transpose_dims=True)

    def mat_det(self, m):
        return np.linalg.det(m.as_np_matrix())

    def mat_norm(self, m):
        return np.linalg.norm(m.nparray)

    def mat_pinv(self, m, rcond):
        return m.np_matrix_transform(
            lambda x: np.linalg.pinv(x, rcond), transpose_dims=True)

    def mat_conj(self, m):
        return m.np_matrix_transform(lambda x: x.conj())

    def mat_n(self, m, prec=None, digits=None):
        # arrays in this base field are already numeric
        return m

    def mat_simplify(self, m, full=False):
        return m

    def mat_expm(self, m, q):
        import scipy.linalg
        return m.np_matrix_transform(lambda x: scipy.linalg.expm(x, q))

    def mat_pow(self, m, n):
        return m.np_matrix_transform(lambda x: x**n)

    def mat_svd(self, m, full_matrices):
        # cast to complex in case we have symbolic vals from Sage
        (u, s, v) = np.linalg.svd(np.matrix(m, dtype=complex), \
            full_matrices=full_matrices)
        return (u, s, v)

    def mat_svd_vals(self, m):
        # cast to complex in case we have symbolic vals from Sage
        (u, s, v) = np.linalg.svd(np.matrix(m, dtype=complex), \
            full_matrices=False)
        return s

    def mat_eig(self, m, w_space, hermit):
        w_space.assert_ket_space()
        eig_fn = np.linalg.eigh if hermit else np.linalg.eig
        # cast to complex in case we have symbolic vals from Sage
        (w, v) = eig_fn(m.as_np_matrix(dtype=complex))

        # sort eigenvalues in ascending order of real component
        srt = np.argsort(-w)
        w = w[srt]
        v = v[:, srt]

        W = (w_space * w_space.H).diag(w)
        V = (m.space.ket_space() * w_space.H).reshaped_np_matrix(v)
        return (W, V)

    def mat_eigvals(self, m, hermit):
        eig_fn = np.linalg.eigvalsh if hermit else np.linalg.eigvals
        # cast to complex in case we have symbolic vals from Sage
        w = eig_fn(m.as_np_matrix(dtype=complex))

        # sort eigenvalues in ascending order of real component
        w = -np.sort(-w)

        if hermit:
            assert np.all(np.imag(w) == 0)
            w = np.real(w)

        return w

    def mat_qr(self, m, inner_space):
        # cast to complex in case we have symbolic vals from Sage
        m_mat = m.as_np_matrix(dtype=complex)
        (q, r) = np.linalg.qr(m_mat)

        if inner_space is None:
            if m_mat.shape[0] < m_mat.shape[1]:
                inner_space = m.space.ket_space()
            else:
                inner_space = m.space.bra_space().H

        inner_space.assert_ket_space()
        
        Q = (m.space.ket_space() * inner_space.H).reshaped_np_matrix(q)
        R = (inner_space * m.space.bra_space()).reshaped_np_matrix(r)

        return (Q, R)

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

        assert isinstance(ket_set, frozenset)
        assert isinstance(bra_set, frozenset)

        for x in ket_set | bra_set:
            self.assert_same(x.base_field)

        # Just return the atoms if possible:
        if len(ket_set) == 1 and len(bra_set) == 0:
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
        """
        return HilbertAtom(label, latex_label, indices, group_op, self)

    def _space_factory(self, ket_set, bra_set):
        r"""
        Factory method for creating ``HilbertSpace`` objects.

        Subclasses can override this method in order to return custom
        subclasses of ``HilbertSpace``.
        """
        return HilbertSpace(ket_set, bra_set, self)

    def _array_factory(self, space, data, noinit_data, reshape, input_axes):
        r"""
        Factory method for creating ``HilbertArray`` objects.

        Subclasses can override this method in order to return custom
        subclasses of ``HilbertArray``.
        """
        return HilbertArray(space, data, noinit_data, reshape, input_axes)

    def indexed_space(self, label, indices, latex_label=None, group_op=None):
        r"""
        Returns a finite-dimensional Hilbert space with an arbitrary index set.

        :param label: a unique label for this Hilbert space
        :param indices: a sequence defining the index set
        :param latex_label: an optional latex representation of the label
        :param group_op: group operation

        ``group_op``, if given, should be a class that defines an
        ``op(self, x, y)`` method.  This supports things like the generalized
        pauliX operator.  The default is ``op(self, x, y) = x*y``.  The
        ``qubit`` and ``qudit`` constructors use ``op(self, x, y) = (x+y)%D``.

        See also: :func:`qitensor.factory.indexed_space`

        >>> from qitensor import base_field_lookup
        >>> field = base_field_lookup(complex)
        >>> ha = field.indexed_space('a', ['x', 'y', 'z'])
        >>> ha
        |a>
        >>> ha.indices
        ['x', 'y', 'z']
        """

        if group_op is None:
            group_op = GroupOpTimes()

        return self._atom_factory(label, latex_label, indices, group_op)

    def qudit(self, label, dim, latex_label=None):
        r"""
        Returns a finite-dimensional Hilbert space with index set [0, 1, ..., n-1].

        :param label: a unique label for this Hilbert space
        :param dim: the dimension of the Hilbert space
        :param latex_label: an optional latex representation of the label

        See also: :func:`qitensor.factory.qudit`

        >>> from qitensor import base_field_lookup
        >>> field = base_field_lookup(complex)
        >>> ha = field.qudit('a', 3)
        >>> ha
        |a>
        >>> ha.indices
        [0, 1, 2]
        """

        group_op = GroupOpCyclic(dim)

        return self.indexed_space(label, range(dim),
            group_op=group_op, latex_label=latex_label)

    def qubit(self, label, latex_label=None):
        r"""
        Returns a two-dimensional Hilbert space with index set [0, 1].

        :param label: a unique label for this Hilbert space
        :param latex_label: an optional latex representation of the label

        See also: :func:`qitensor.factory.qubit`

        >>> from qitensor import base_field_lookup
        >>> field = base_field_lookup(complex)
        >>> ha = field.qubit('a')
        >>> ha
        |a>
        >>> ha.indices
        [0, 1]
        """

        return self.qudit(label, 2, latex_label)
