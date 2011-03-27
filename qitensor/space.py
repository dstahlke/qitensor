"""
HilbertSpaces are built of HilbertAtom's.  They represent the spaces that
HilbertArray's live in.  A HilbertSpace is typically created by applying the
multiplication operator to HilbertAtom's or other HilbertSpace's.
"""

import numpy as np

from qitensor import have_sage, shape_product
from qitensor.exceptions import *

__all__ = ['HilbertSpace']

class HilbertSpace(object):
    def __init__(self, ket_set, bra_set, base_field):
        """
        Constructor should only be called from :meth:`HilbertBaseField._space_factory`
        """

        self.base_field = base_field
        self._H = None

        # If ket_set is None then we are being called from the HilbertAtom
        # subclass constructor.  That constructor will take care of setting up
        # these attributes.
        if not ket_set is None:
            for x in ket_set:
                assert not x.is_dual
            for x in bra_set:
                assert x.is_dual

            self.bra_ket_set = bra_set | ket_set
            self.ket_set = ket_set
            self.bra_set = bra_set
            self.sorted_kets = sorted([x for x in ket_set])
            self.sorted_bras = sorted([x for x in bra_set])

            # Make sure all atoms are compatible, otherwise raise
            # a MismatchedIndexSetError
            sorted([x for x in self.bra_ket_set])
            
            ket_shape = [len(x.indices) for x in self.sorted_kets]
            bra_shape = [len(x.indices) for x in self.sorted_bras]
            self.shape = tuple(ket_shape + bra_shape)

    def bra_space(self):
        """
        Returns a ``HilbertSpace`` consisting of only the bra space of this
        space.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> sp = (ha.H * hb * hc * hc.H); sp
        |b,c><a,c|
        >>> sp.bra_space()
        <a,c|
        """
        return self.base_field.create_space2(frozenset(), self.bra_set)

    def ket_space(self):
        """
        Returns a ``HilbertSpace`` consisting of only the ket space of this
        space.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> sp = (ha.H * hb * hc * hc.H); sp
        |b,c><a,c|
        >>> sp.ket_space()
        |b,c>
        """
        return self.base_field.create_space2(self.ket_set, frozenset())

    @property
    def H(self):
        """
        The adjoint of this Hilbert space.

        A ``HilbertSpace`` is returned which has bras turned into kets and
        kets turned into bras.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> sp = ha * hb * hc.H; sp
        |a,b><c|
        >>> sp.H
        |c><a,b|
        >>> sp.H.H
        |a,b><c|
        """
        if self._H is None:
            self._H = self.base_field.create_space1(
                [x.H for x in self.bra_ket_set])
        return self._H

    @property
    def O(self):
        """
        The operator space for a bra or ket space.

        This just returns ``self * self.H``.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> ha.O
        |a><a|
        >>> (ha*hb).O
        |a,b><a,b|
        """
        return self * self.H

    def __cmp__(self, other):
        if self.sorted_kets < other.sorted_kets:
            return -1
        elif self.sorted_kets > other.sorted_kets:
            return 1
        if self.sorted_bras < other.sorted_bras:
            return -1
        elif self.sorted_bras > other.sorted_bras:
            return 1
        else:
            return 0

    def __hash__(self):
        if len(self.ket_set) + len(self.bra_set) == 1:
            # in this case, HilbertAtom.__hash__ should have been called instead
            raise Exception()
        else:
            return hash(self.ket_set) ^ hash(self.bra_set)

    def __str__(self):
        bra_labels = [x.label for x in self.sorted_bras]
        ket_labels = [x.label for x in self.sorted_kets]
        if len(ket_labels) > 0 and len(bra_labels) > 0:
            return '|'+(','.join(ket_labels))+'><'+ \
                (','.join(bra_labels))+'|'
        elif len(ket_labels) > 0:
            return '|'+(','.join(ket_labels))+'>'
        elif len(bra_labels) > 0:
            return '<'+(','.join(bra_labels))+'|'
        else:
            return '|>'

    def __repr__(self):
        return str(self)

    def _latex_(self): # for Sage
        bra_labels = [x.latex_label for x in self.sorted_bras]
        ket_labels = [x.latex_label for x in self.sorted_kets]
        if len(ket_labels) > 0 and len(bra_labels) > 0:
            return '\\left| '+(','.join(ket_labels))+ \
                ' \\right\\rangle\\left\\langle '+ \
                (','.join(bra_labels))+' \\right|'
        elif len(ket_labels) > 0:
            return '\\left| '+(','.join(ket_labels))+' \\right\\rangle'
        elif len(bra_labels) > 0:
            return '\\left\\langle '+(','.join(bra_labels))+' \\right|'
        else:
            return '\\left|\\right\\rangle'

    def __mul__(self, other):
        if not isinstance(other, HilbertSpace):
            raise TypeError('HilbertSpace can only multiply HilbertSpace')
        self.base_field.assert_same(other.base_field)

        common_kets = self.ket_set & other.ket_set
        common_bras = self.bra_set & other.bra_set
        if common_kets or common_bras:
            raise DuplicatedSpaceError(
                self.base_field.create_space2(common_kets, common_bras))
        return self.base_field.create_space1(
            self.bra_ket_set | other.bra_ket_set)

    def __rmul__(self, other):
        return self.__mul__(other)

    def diag(self, v):
        """
        Create a diagonal operator from the given 1-d list.

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)

        >>> ha.diag([1, 2])
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j,  2.+0.j]]))
        >>> ha.diag([1, 2]) == ha.H.diag([1, 2])
        True
        >>> ha.diag([1, 2]) == ha.O.diag([1, 2])
        True

        >>> op =  (ha*hb).diag([1, 2, 3, 4, 5, 6])
        >>> # NOTE: spaces are ordered lexicographically
        >>> op == (hb*ha).diag([1, 2, 3, 4, 5, 6])
        True
        >>> op.space
        |a,b><a,b|
        >>> import numpy
        >>> numpy.diag( op.as_np_matrix() )     
        array([ 1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  5.+0.j,  6.+0.j])
        """

        if len(self.ket_set) == 0 or len(self.bra_set) == 0:
            return self.O.diag(v)

        diag = np.diagflat(v)
        return self.reshaped_np_matrix(diag)

    def reshaped_np_matrix(self, m):
        """
        Returns a ``HilbertArray`` created from a given numpy matrix.

        The number of rows and columns must match the dimensions of the ket and
        bra spaces.  It is required that ``len(m.shape)==2``.

        :param m: the input matrix.
        :type m: numpy.matrix or numpy.array

        See also: :func:`array`

        >>> import numpy
        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')

        >>> ha.reshaped_np_matrix(numpy.matrix([[1], [2]]))
        HilbertArray(|a>,
        array([ 1.+0.j,  2.+0.j]))

        >>> ha.H.reshaped_np_matrix(numpy.matrix([1, 2]))
        HilbertArray(<a|,
        array([ 1.+0.j,  2.+0.j]))

        >>> d = (ha*hb).O.reshaped_np_matrix(numpy.diag([1, 2, 3, 4]))
        >>> d == (ha*hb).diag([1, 2, 3, 4])
        True
        >>> d
        HilbertArray(|a,b><a,b|,
        array([[[[ 1.+0.j,  0.+0.j],
                 [ 0.+0.j,  0.+0.j]],
        <BLANKLINE>
                [[ 0.+0.j,  2.+0.j],
                 [ 0.+0.j,  0.+0.j]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 0.+0.j,  0.+0.j],
                 [ 3.+0.j,  0.+0.j]],
        <BLANKLINE>
                [[ 0.+0.j,  0.+0.j],
                 [ 0.+0.j,  4.+0.j]]]]))
        """

        ket_size = shape_product([len(x.indices) for x in self.ket_set])
        bra_size = shape_product([len(x.indices) for x in self.bra_set])

        if len(m.shape) != 2 or m.shape[0] != ket_size or m.shape[1] != bra_size:
            raise HilbertShapeError(m.shape, (ket_size, bra_size))

        return self.array(m, reshape=True)

    def array(self, data=None, noinit_data=False, reshape=False):
        """
        Returns a ``HilbertArray`` created from the given data, or filled with
        zeros if no data is given.

        If the ``reshape`` parameter is ``True`` then the given ``data`` array
        can be any shape as long as the total number of elements is equal to
        the dimension of this Hilbert space (bra dimension times ket
        dimension).  If ``reshape`` is ``False`` (the default) then ``data``
        must have an axis for each of the components of this Hilbert space.

        Since it is not always clear which axes should correspond to which
        Hilbert space components, it is recommended to specify ``data`` only
        when there is at most one bra component and at most one ket component.
        In this case, the first axis will correspond to the ket space (if it
        exists) and the last axis will correspond ot the bra space (if it
        exists).

        :param data: the array will be initialized with the data if given,
            otherwise it will be initialized with zeros.
        :type data: anything that can be used to create a numpy.array

        :param noinit_data: If true, the underlying numpy.array object will not
            be allocated.  Don't use this unless you are sure you know what you
            are doing.
        :type noinit_data: bool; default False

        :param reshape: If true, the array given by the ``data`` parameter will
            be reshaped if needed.  Otherwise, an exception will be raised if
            it is not the proper shape.
        :type reshape: bool; default False

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.array([1,2])
        HilbertArray(|a>,
        array([ 1.+0.j,  2.+0.j]))

        >>> ha.H.array([1,2])
        HilbertArray(<a|,
        array([ 1.+0.j,  2.+0.j]))

        >>> ha.O.array([[1, 2], [3, 4]])
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  2.+0.j],
               [ 3.+0.j,  4.+0.j]]))
        
        >>> ha.O.array([1, 2, 3, 4], reshape=True)
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  2.+0.j],
               [ 3.+0.j,  4.+0.j]]))
        """

        return self.base_field._array_factory(self, data, noinit_data, reshape)

    def random_array(self):
        """
        Returns a ``HilbertArray`` with random values.

        The real and complex components each have uniform distribution in the
        ``[-1, 1]`` range.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.random_array() # doctest: +SKIP
        HilbertArray(|a>,
        array([-0.484410+0.426767j,  0.000693+0.912554j]))

        """
        return self.array(self.base_field.random_array(self.shape))

    def eye(self):
        """
        Returns a ``HilbertArray`` corresponding to the identity matrix.

        If the bra space or ket space is empty, then the nonempty of those two
        is used to form an operator space (i.e. ``self.O``).  If both the
        bra and ket spaces are nonempty, they must be of the same dimension
        since the identity matrix must be square.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')

        >>> ha.eye()
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j,  1.+0.j]]))

        >>> (ha*hb.H).eye()
        HilbertArray(|a><b|,
        array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j,  1.+0.j]]))

        >>> ha.eye() == ha.H.eye()
        True
        >>> ha.eye() == ha.O.eye()
        True
        """

        if len(self.ket_set) == 0 or len(self.bra_set) == 0:
            return (self * self.H).eye()

        ket_size = shape_product([len(x.indices) for x in self.ket_set])
        bra_size = shape_product([len(x.indices) for x in self.bra_set])
        if bra_size != ket_size:
            raise HilbertShapeError(bra_size, ket_size)

        return self.array(self.base_field.eye(bra_size), reshape=True)

    def basis_vec(self, idx):
        """
        Returns a ``HilbertArray`` corresponding to a basis vector.

        The returned vector has a 1 in the slot corresponding to ``idx`` and
        zeros elsewhere.

        :param idx: if this Hilbert space has only one component, this should
            be a member of that component's index set.  Otherwise, this should
            be a tuple of indices.

        >>> from qitensor import qubit, indexed_space
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = indexed_space('c', ['x', 'y', 'z'])

        >>> ha.basis_vec(0)
        HilbertArray(|a>,
        array([ 1.+0.j,  0.+0.j]))

        >>> ha.basis_vec(1)
        HilbertArray(|a>,
        array([ 0.+0.j,  1.+0.j]))

        >>> (ha*hb).basis_vec((0, 1))
        HilbertArray(|a,b>,
        array([[ 0.+0.j,  1.+0.j],
               [ 0.+0.j,  0.+0.j]]))

        >>> hc.basis_vec('y')
        HilbertArray(|c>,
        array([ 0.+0.j,  1.+0.j,  0.+0.j]))
        """

        ret = self.array()
        index_map = ret._index_key_to_map(idx)
        if len(index_map) != len(self.shape):
            raise HilbertIndexError('not enough indices given')
        ret[idx] = 1
        return ret

    def assert_ket_space(self):
        """
        Throws an exception unless the bra space is empty.
        """

        if self.bra_set:
            raise NotKetSpaceError(repr(self))

    ########## stuff that only works in Sage ##########

    def reshaped_sage_matrix(self, m):
        if not have_sage:
            raise HilbertError('This is only available under Sage')

        sage_ring = self.base_field.sage_ring
        if m.base_ring() != sage_ring:
            m = m.change_ring(sage_ring)
        return self.reshaped_np_matrix(np.array(m, dtype=self.base_field.dtype))

    ########## end of stuff that only works in Sage ##########
