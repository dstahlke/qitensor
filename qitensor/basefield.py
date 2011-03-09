import numpy as np
import numpy.random
import numpy.linalg as linalg

from qitensor.exceptions import *
from qitensor.atom import HilbertAtom
from qitensor.array import HilbertArray
from qitensor.space import HilbertSpace

__all__ = ['HilbertBaseField']

class HilbertBaseField(object):
    def __init__(self, dtype, unique_id):
        self.dtype = dtype
        self.unique_id = unique_id

    def assert_same(self, other):
        if self.unique_id != other.unique_id:
            raise IncompatibleBaseFieldError('Different base_fields: '+
                repr(self)+' vs. '+repr(other))

    def complex_unit(self):
        return 1j

    def fractional_phase(self, a, b):
        return np.exp(2j * np.pi * a / b)

    def random_array(self, shape):
        return numpy.random.random((shape))

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
        return linalg.det(m.as_np_matrix())

    def mat_norm(self, m):
        return linalg.norm(m.nparray)

    def mat_pinv(self, m, rcond):
        return m.np_matrix_transform(
            lambda x: np.linalg.pinv(x, rcond), transpose_dims=True)

    def mat_conj(self, m):
        return m.np_matrix_transform(lambda x: x.conj())

    def mat_expm(self, m, q):
        import scipy.linalg
        return m.np_matrix_transform(lambda x: scipy.linalg.expm(x, q))

    def mat_svd_full(self, m, s_space):
        (u, s, v) = linalg.svd(m.as_np_matrix())
        u_space = m.space.ket_space() * s_space.ket_space().H
        v_space = m.space.bra_space() * s_space.bra_space().H
        U = u_space.reshaped_np_matrix(u)
        V = v_space.reshaped_np_matrix(v)

        dim1 = np.product(s_space.ket_space().shape)
        dim2 = np.product(s_space.bra_space().shape)
        min_dim = np.min([dim1, dim2])
        Sm = np.zeros((dim1, dim2), dtype=self.dtype)
        Sm[:min_dim, :min_dim] = np.diag(s)
        S = s_space.reshaped_np_matrix(Sm)

        return (U, S, V)

    def mat_svd_partial(self, m, s_space):
        (u, s, v) = linalg.svd(m.as_np_matrix(), full_matrices=False)
        u_space = m.space.ket_space() * s_space.H
        s_mat_space = s_space * s_space.H
        v_space = s_space * m.space.bra_space()
        U = u_space.reshaped_np_matrix(u)
        V = v_space.reshaped_np_matrix(v)
        S = s_mat_space.reshaped_np_matrix(np.diag(s))

        return (U, S, V)

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
            if x.base_field != self:
                raise IncompatibleBaseFieldError('Different base_fields: '+
                    repr(self)+' vs. '+repr(x.base_field))

        # Just return the atoms if possible:
        if len(ket_set) == 1 and len(bra_set) == 0:
            return list(ket_set)[0]
        elif len(ket_set) == 0 and len(bra_set) == 1:
            return list(bra_set)[0]
        else:
            return self._space_factory(ket_set, bra_set)

    def _atom_factory(self, label, latex_label, indices):
        r"""
        Factory method for creating ``HilbertAtom`` objects.

        Subclasses can override this method in order to return custom
        subclasses of ``HilbertAtom``.
        """
        return HilbertAtom(label, latex_label, indices, self)

    def _space_factory(self, ket_set, bra_set):
        r"""
        Factory method for creating ``HilbertSpace`` objects.

        Subclasses can override this method in order to return custom
        subclasses of ``HilbertSpace``.
        """
        return HilbertSpace(ket_set, bra_set, self)

    def _array_factory(self, space, data, noinit_data, reshape):
        r"""
        Factory method for creating ``HilbertArray`` objects.

        Subclasses can override this method in order to return custom
        subclasses of ``HilbertArray``.
        """
        return HilbertArray(space, data, noinit_data, reshape)

    def indexed_space(self, label, indices, latex_label=None):
        r"""
        Returns a finite-dimensional Hilbert space with an arbitrary index set.

        :param label: a unique label for this Hilbert space
        :param indices: a sequence defining the index set
        :param latex_label: an optional latex representation of the label

        See also: :func:`qitensor.functions.indexed_space`

        >>> from qitensor import base_field_lookup
        >>> field = base_field_lookup(complex)
        >>> ha = field.indexed_space('a', ['x', 'y', 'z'])
        >>> ha
        |a>
        >>> ha.indices
        ['x', 'y', 'z']
        """

        return self._atom_factory(label, latex_label, indices)

    def qudit(self, label, dim, latex_label=None):
        r"""
        Returns a finite-dimensional Hilbert space with index set [0, 1, ..., n-1].

        :param label: a unique label for this Hilbert space
        :param dim: the dimension of the Hilbert space
        :param latex_label: an optional latex representation of the label

        See also: :func:`qitensor.functions.qudit`

        >>> from qitensor import base_field_lookup
        >>> field = base_field_lookup(complex)
        >>> ha = field.qudit('a', 3)
        >>> ha
        |a>
        >>> ha.indices
        [0, 1, 2]
        """

        return self.indexed_space(label, range(dim), latex_label)

    def qubit(self, label, latex_label=None):
        r"""
        Returns a two-dimensional Hilbert space with index set [0, 1].

        :param label: a unique label for this Hilbert space
        :param latex_label: an optional latex representation of the label

        See also: :func:`qitensor.functions.qubit`

        >>> from qitensor import base_field_lookup
        >>> field = base_field_lookup(complex)
        >>> ha = field.qubit('a')
        >>> ha
        |a>
        >>> ha.indices
        [0, 1]
        """

        return self.qudit(label, 2, latex_label)
