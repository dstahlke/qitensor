"""
HilbertSpaces are tensor products of HilbertAtoms (although individual
HilbertAtoms are also HilbertSpaces).  They represent the spaces that
HilbertArrays live in.  A HilbertSpace is typically created by applying the
multiplication operator to HilbertAtoms or other HilbertSpaces.
"""

import numpy as np
import itertools
import operator

#import weakref

from qitensor import have_sage
from qitensor.exceptions import DuplicatedSpaceError, HilbertError, \
    MismatchedSpaceError, HilbertShapeError, NotKetSpaceError
import qitensor.atom
from qitensor.arrayformatter import FORMATTER
from qitensor.subspace import TensorSubspace
from qitensor.array import HilbertArray
from qitensor.array cimport HilbertArray

__all__ = ['HilbertSpace']

# helper function for hadamard
cdef int _countbits(int i):
    """
    Count the number of set bits.

    >>> _countbits(0xfeedf00d)
    FIXME - doctest doesn't run?
    """

    cdef int n = 0
    while i:
        n += i & 1
        i >>= 1
    return n

# helper function for hadamard
cdef int _int_log2(int i):
    """
    Returns the log2 of the input if input is a power of 2, otherwise returns -1.

    >>> [_int_log2(i) for i in (0, 1, 32, 33)]
    FIXME - doctest doesn't run?
    """

    cdef int n = 0
    while not i & 1:
        i >>= 1
        n += 1
    if i == 1:
        return n
    else:
        return -1

def _unreduce_v1(ket_set, bra_set):
    """
    This is the function that handles restoring a pickle.
    """

    base_field = list(ket_set | bra_set)[0].base_field
    return _space_factory(ket_set, bra_set)

#_space_cache = weakref.WeakValueDictionary()
cdef dict _space_cache = dict()

cpdef _space_factory(frozenset ket_set, frozenset bra_set):
    r"""
    Factory method for creating ``HilbertSpace`` objects.

    Subclasses can override this method in order to return custom
    subclasses of ``HilbertSpace``.

    Users shouldn't call this function.
    """

    cdef tuple key = (ket_set, bra_set)

    if not _space_cache.has_key(key):
        spc = HilbertSpace(ket_set, bra_set)
        _space_cache[key] = spc

    return _space_cache[key]

cpdef create_space1(kets_and_bras):
    r"""
    Creates a ``HilbertSpace`` from a collection of ``HilbertAtom`` objects.

    This provides an alternative to using the multiplication operator
    to combine ``HilbertAtom`` objects.

    :param kets_and_bras: a collection of ``HilbertAtom`` objects

    >>> from qitensor import qubit
    >>> ha = qubit('a')
    >>> hb = qubit('b')
    >>> ha * hb == create_space1([ha, hb])
    True
    >>> ha.H * hb == create_space1([ha.H, hb])
    True
    """
    return create_space2(
        frozenset([x for x in kets_and_bras if not x.is_dual]),
        frozenset([x for x in kets_and_bras if x.is_dual]))

cpdef create_space2(frozenset ket_set, frozenset bra_set):
    r"""
    Creates a ``HilbertSpace`` from frozensets of ``HilbertAtom`` kets and bras.

    This provides an alternative to using the multiplication operator
    to combine ``HilbertAtom`` objects.

    :param ket_set: a collection of ``HilbertAtom`` objects for which ``is_dual==False``
    :param bra_set: a collection of ``HilbertAtom`` objects for which ``is_dual==True``

    >>> from qitensor import qubit
    >>> ha = qubit('a')
    >>> hb = qubit('b')
    >>> ha * hb == create_space2(frozenset([ha, hb]), frozenset())
    True
    >>> ha.H * hb == create_space2(frozenset([hb]), frozenset([ha.H]))
    True
    """

    # Just return the atoms if possible:
    if not ket_set and not bra_set:
        raise HilbertError('tried to create empty HilbertSpace')
    elif len(ket_set) == 1 and not bra_set:
        return ket_set.__iter__().next()
    elif not ket_set and len(bra_set) == 1:
        return bra_set.__iter__().next()
    else:
        return _space_factory(ket_set, bra_set)

cpdef long _shape_product(l):
    """
    Multiplies a tuple of integers together.

    Used to convert shape to dimension.

    >>> from qitensor.space import _shape_product
    >>> _shape_product((1,2,3,4))
    24
    """

    # faster than np.prod(l, dtype=int)
    return reduce(operator.mul, l, 1)

########################################

cdef class HilbertSpace:
    def __init__(self, ket_set, bra_set, _H=None):
        """
        Constructor should only be called from :meth:`_space_factory`.

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a'); ha
        |a>
        >>> hb = qudit('b', 3); hb
        |b>
        >>> ha*hb
        |a,b>

        sage: from qitensor import qudit
        sage: ha = qudit('a', 3)
        sage: hb = qudit('b', 5)
        sage: ha*hb
        |a,b>
        sage: TestSuite(ha*hb).run()
        """

        self._H = _H
        self._prime = None

        # (Sphinx docstrings)
        #: In the case of direct sum spaces, this is a list of the components.
        self.addends = None

        assert isinstance(ket_set, frozenset)
        assert isinstance(bra_set, frozenset)

        for x in ket_set:
            assert not x.is_dual
        for x in bra_set:
            assert x.is_dual

        #: A frozenset consisting of the ket atoms that this space is made of.
        self.ket_set = ket_set
        #: A frozenset consisting of the bra atoms that this space is made of.
        self.bra_set = bra_set
        #: A frozenset consisting of the union of ``self.bra_set`` and ``self.ket_set``.
        self.bra_ket_set = bra_set | ket_set
        #: A sorted list consisting of the ket atoms that this space is made of.
        self.sorted_kets = sorted(list(ket_set))
        #: A sorted list consisting of the bra atoms that this space is made of.
        self.sorted_bras = sorted(list(bra_set))

        if len(self.bra_ket_set) == 0:
            raise HilbertError('tried to create empty HilbertSpace')

        #: The HilbertBaseField that defines the numerical properties of arrays belonging
        #: to this space.
        self.base_field = list(self.bra_ket_set)[0].base_field

        # Make sure all atoms are compatible, otherwise raise
        # a MismatchedSpaceError
        qitensor.atom._assert_all_compatible(self.bra_ket_set)

        for x in self.bra_ket_set:
            self.base_field.assert_same(x.base_field)
        
        ket_shape = [len(x.indices) for x in self.sorted_kets]
        bra_shape = [len(x.indices) for x in self.sorted_bras]

        #: A tuple consisting of the dimensions of the underlying atoms that make up this space.
        self.shape = tuple(ket_shape + bra_shape)
        self._dim = _shape_product(self.shape)
        self._is_simple_dyad = len(bra_set)==1 and len(ket_set)==1

        self.axes = self.sorted_kets + self.sorted_bras
        self.axes_lookup = dict((s, self.axes.index(s)) for s in self.axes)

    def __reduce__(self):
        """
        Tells pickle how to store this object.
        """
        return _unreduce_v1, (self.ket_set, self.bra_set)

    @classmethod
    def _expand_list_to_atoms(cls, list_in):
        """
        Expands a list of HilbertSpaces to a list of HilbertAtoms.

        >>> from qitensor import qubit, HilbertSpace
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> HilbertSpace._expand_list_to_atoms([ha, ha*hb.H*hc, ha])
        [|a>, |a>, |c>, <b|, |a>]
        """

        list_out = []
        for x in list_in:
            assert isinstance(x, HilbertSpace)
            list_out += sorted(x.ket_set)
            list_out += sorted(x.bra_set)
        return list_out

    @classmethod
    def _assert_nodup_space(cls, spaces, msg):
        """
        Raises a DuplicatedSpaceError if any of the given spaces share a common
        HilbertAtom.

        >>> from qitensor import qubit, HilbertSpace
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> HilbertSpace._assert_nodup_space([ha, hb*ha.H, hc.O], 'oops')
        >>> HilbertSpace._assert_nodup_space([ha, hc*ha.H, hc.O], 'oops')
        Traceback (most recent call last):
            ...
        DuplicatedSpaceError: 'oops: |c>'
        """

        # FIXME - use this function more often

        cdef set seen = set()
        cdef set dupes = set()
        for s in cls._expand_list_to_atoms(spaces):
            assert isinstance(s, qitensor.atom.HilbertAtom)
            if s in seen:
                dupes.add(s)
            seen.add(s)
        if dupes:
            common_kets = frozenset(x for x in dupes if not x.is_dual)
            common_bras = frozenset(x for x in dupes if x.is_dual)
            spc = create_space2(common_kets, common_bras)
            raise DuplicatedSpaceError(spc, msg)

    cpdef HilbertSpace bra_space(self):
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
        return create_space2(frozenset(), self.bra_set)

    cpdef HilbertSpace ket_space(self):
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
        return create_space2(self.ket_set, frozenset())

    cpdef is_symmetric(self):
        """
        Check whether the bra and ket spaces are the same.
        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> ha.is_symmetric()
        False
        >>> ha.O.is_symmetric()
        True
        >>> (ha * hb.H).is_symmetric()
        False
        >>> (ha * hb.H * ha.H * hb).is_symmetric()
        True
        """
        return self == self.H

    cpdef is_square(self):
        """
        If the dimension of the bra and ket spaces are equal, returns this
        common dimension.  Otherwise returns zero.

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> ha.is_square()
        0
        >>> ha.O.is_square()
        2
        >>> (ha*hb.H).is_square()
        0
        >>> (ha*hb).O.is_square()
        6
        """

        ket_size = _shape_product([len(x.indices) for x in self.ket_set])
        bra_size = _shape_product([len(x.indices) for x in self.bra_set])

        if bra_size == ket_size:
            return bra_size
        else:
            return 0

    cpdef assert_square(self):
        """
        If the dimension of the bra and ket spaces are equal, returns this
        common dimension.  Otherwise throws a HilbertShapeError.

        >>> from qitensor import qudit
        >>> qudit('a', 3).assert_square()
        Traceback (most recent call last):
            ...
        HilbertShapeError: '3 vs. 1'
        >>> (qudit('a', 3) * qudit('b', 4).H).assert_square()
        Traceback (most recent call last):
            ...
        HilbertShapeError: '3 vs. 4'
        """

        ket_size = _shape_product([len(x.indices) for x in self.ket_set])
        bra_size = _shape_product([len(x.indices) for x in self.bra_set])

        if bra_size == ket_size:
            return bra_size
        else:
            raise HilbertShapeError(ket_size, bra_size)

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
            self._H = create_space1(
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

    @property
    def prime(self):
        """
        Returns a ``HilbertSpace`` just like this one but with an apostrophe
        appended to each label.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> (ha*hb).prime
        |a',b'>
        >>> (ha*hb).prime.prime
        |a'',b''>
        >>> ha.O.prime
        |a'><a'|
        """

        if self._prime is None:
            self._prime = create_space1([ x.prime for x in self.bra_ket_set ])
        return self._prime

    def __richcmp__(self, other, op):
        """
        Compares two HilbertSpace objects lexicographically.
        """

        if not isinstance(other, qitensor.space.HilbertSpace):
            if op == 2: # ==
                return False
            elif op == 3: # !=
                return True
            else:
                assert isinstance(other, HilbertSpace)

        eq = (self.sorted_kets == other.sorted_kets) and \
            (self.sorted_bras == other.sorted_bras)

        if op == 0 or op == 1: # < or <=
            if self.sorted_kets < other.sorted_kets:
                lt = True
            elif self.sorted_kets > other.sorted_kets:
                lt = False
            if self.sorted_bras < other.sorted_bras:
                lt = True
            else:
                lt = False
            return lt if op==0 else (lt or eq)
        elif op == 2 or op == 3: # == or !=
            return eq if op==2 else not eq
        elif op == 4 or op == 5: # > or >=
            if self.sorted_kets > other.sorted_kets:
                gt = True
            elif self.sorted_kets < other.sorted_kets:
                gt = False
            if self.sorted_bras > other.sorted_bras:
                gt = True
            else:
                gt = False
            return gt if op==4 else (gt or eq)

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

    def _repr_latex_(self):
        """
        Returns a latex representation, for IPython.
        """

        if not FORMATTER.ipy_space_format_mode == 'latex':
            return None
        return '$'+self._latex_()+'$'

    def _repr_png_(self):
        """
        Returns a PNG representation, for IPython.
        """

        if not FORMATTER.ipy_space_format_mode == 'png':
            return None

        # the following is adapted from sympyprint.py
        from IPython.lib.latextools import latex_to_png
        s = self._latex_()
        # As matplotlib does not support display style, dvipng backend is used here.
        png = latex_to_png(s, backend='dvipng', wrap=True)
        return png

    def _latex_(self):
        """
        Returns a latex representation, for Sage.
        """

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
        if not isinstance(self, HilbertSpace) or not isinstance(other, HilbertSpace):
            return NotImplemented

        self.base_field.assert_same(other.base_field)

        common_kets = self.ket_set & other.ket_set
        common_bras = self.bra_set & other.bra_set
        if common_kets or common_bras:
            raise DuplicatedSpaceError(
                create_space2(common_kets, common_bras))
        return create_space1(
            self.bra_ket_set | other.bra_ket_set)

    def __div__(self, other):
        """
        Returns a HilbertSpace ``ret`` with the property that ``other*ret==self``.
        An error is thrown if such a relation is not possible.
        """

        if not isinstance(self, HilbertSpace) or not isinstance(other, HilbertSpace):
            return NotImplemented

        if other.bra_ket_set == self.bra_ket_set:
            raise MismatchedSpaceError("dividing "+repr(self)+" by itself would result in 1-dimensional space")
        if not other.bra_ket_set < self.bra_ket_set:
            raise MismatchedSpaceError(repr(self)+" doesn't contain "+repr(other))
        return create_space1(self.bra_ket_set - other.bra_ket_set)

    def __truediv__(self, other):
        """
        Returns a HilbertSpace ``ret`` with the property that ``other*ret==self``.
        An error is thrown if such a relation is not possible.
        """

        return self.__div__(other)

    cpdef HilbertArray diag(self, v):
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

    cpdef reshaped_np_matrix(self, m, input_axes=None):
        """
        Returns a ``HilbertArray`` created from a given numpy matrix.

        The number of rows and columns must match the dimensions of the ket and
        bra spaces.  It is required that ``len(m.shape)==2``.  The input_axes
        parameter gives the storage order of the input data, and is recommended
        when the input spaces are composites (not HilbertAtoms).

        :param m: the input matrix.
        :type m: numpy.matrix or numpy.array
        :param input_axes: the storage order for the input data
        :type input_axes: list

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

        >>> (ha.H*hb.H).reshaped_np_matrix(numpy.array([[1, 2, 3, 4]]), input_axes=[ha.H, hb.H])
        HilbertArray(<a,b|,
        array([[ 1.+0.j,  2.+0.j],
               [ 3.+0.j,  4.+0.j]]))
        >>> (ha.H*hb.H).reshaped_np_matrix(numpy.array([[1, 2, 3, 4]]), input_axes=[hb.H, ha.H])
        HilbertArray(<a,b|,
        array([[ 1.+0.j,  3.+0.j],
               [ 2.+0.j,  4.+0.j]]))
        """

        ket_size = _shape_product([len(x.indices) for x in self.ket_set])
        bra_size = _shape_product([len(x.indices) for x in self.bra_set])

        if len(m.shape) != 2 or m.shape[0] != ket_size or m.shape[1] != bra_size:
            raise HilbertShapeError(m.shape, (ket_size, bra_size))

        return self.array(m, reshape=True, input_axes=input_axes)

    cpdef array(self, data=None, cpython.bool noinit_data=False, cpython.bool reshape=False, input_axes=None):
        """
        Returns a ``HilbertArray`` created from the given data, or filled with
        zeros if no data is given.

        If the ``reshape`` parameter is ``True`` then the given ``data`` array
        can be any shape as long as the total number of elements is equal to
        the dimension of this Hilbert space (bra dimension times ket
        dimension).  If ``reshape`` is ``False`` (the default) then ``data``
        must have an axis for each of the components of this Hilbert space.

        Since it is not always clear which axes should correspond to which
        Hilbert space components, it is recommended when using the ``data``
        parameter to also specify ``input_axes`` to tell which HilbertAtom maps
        to which axis of the input array.  Note that there is no ambiguity if
        the input and output spaces are both HilbertAtoms (not composite
        spaces): In this case, the first axis will correspond to the ket space
        (if it exists) and the last axis will correspond ot the bra space (if
        it exists).

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

        :param input_axes: Tells how the axes map to the space.  Default is
            lexicographically based upon the names of the HilbertAtoms (it
            is not recommended to rely on this ordering).
        :type input_axes: tuple of HilbertAtoms

        >>> from qitensor import qubit, qudit
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
        
        >>> import numpy
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> hc = qudit('c', 4)
        >>> arr = numpy.zeros((2, 3, 4))
        >>> arr[1,0,0] = 1
        >>> arr[0,1,0] = 2
        >>> arr[0,0,1] = 3
        >>> x = (ha*hb.H*hc).array(arr, input_axes=(ha, hb.H, hc))
        >>> x.space
        |a,c><b|
        >>> x.nparray.shape
        (2, 4, 3)
        >>> x[{ ha: 1, hb.H: 0, hc: 0 }]
        (1+0j)
        >>> x[{ ha: 0, hb.H: 1, hc: 0 }]
        (2+0j)
        >>> x[{ ha: 0, hb.H: 0, hc: 1 }]
        (3+0j)
        """

        return HilbertArray(self, data, noinit_data, reshape, input_axes)

    cpdef HilbertArray random_array(self):
        """
        Returns a ``HilbertArray`` with random values.

        The values are complex numbers drawn from a standard normal distribution.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.random_array() # doctest: +SKIP
        HilbertArray(|a>,
        array([-0.484410+0.426767j,  0.000693+0.912554j]))

        """
        return self.array(self.base_field.random_array(self.shape))

    cpdef HilbertArray random_unitary(self):
        """
        Returns a random unitary.

        If the bra space or ket space is empty, then the nonempty of those two
        is used to form an operator space (i.e. ``self.O``).  If both the
        bra and ket spaces are nonempty, they must be of the same dimension
        since a unitary matrix must be square.

        The returned unitary is drawn from a distribution uniform with respect
        to the Haar measure, using the algorithm by Mezzadri, "How to Generate
        Random Matrices from the Classical Compact Groups", Notices of the AMS
        54, 592 (2007).

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> m = ha.random_unitary()
        >>> (m.H * m - ha.eye()).norm() < 1e-14
        True
        >>> (m * m.H - ha.eye()).norm() < 1e-14
        True
        >>> (ha.random_unitary() - ha.random_unitary()).norm() < 1e-14
        False
        >>> m = (ha.H * hb).random_unitary()
        >>> m.space
        |b><a|
        >>> (m.H * m - ha.eye()).norm() < 1e-14
        True
        >>> (m * m.H - hb.eye()).norm() < 1e-14
        True
        """

        if len(self.ket_set) == 0 or len(self.bra_set) == 0:
            return (self * self.H).random_unitary()

        self.assert_square()

        z = self.random_array().as_np_matrix()
        (q, r) = self.base_field.mat_qr(z)
        d = np.diag(r)
        ph = d/np.abs(d)
        ret = np.multiply(q, ph)
        return self.reshaped_np_matrix(ret)

    cpdef HilbertArray random_isometry(self):
        """
        Returns a random isometry.

        The ket space must be at least as great in dimension as the bra space.

        >>> from qitensor import qubit, qudit

        >>> ha = qubit('a')
        >>> hb = qudit('b', 7)
        >>> hc = qudit('c', 3)
        >>> m = (ha * hb * hc.H).random_isometry()
        >>> (m.H * m - hc.eye()).norm() < 1e-14
        True
        >>> m = (hb * ha.H * hc.H).random_isometry()
        >>> (m.H * m - (ha*hc).eye()).norm() < 1e-14
        True
        """

        if len(self.ket_set) == 0 or len(self.bra_set) == 0:
            raise HilbertError('not an operator space: '+str(self))

        dk = self.ket_space().dim()
        db = self.bra_space().dim()
        if dk < db:
            raise HilbertShapeError(dk, db)

        U = self.ket_space().O.random_unitary()
        iso = U.as_np_matrix()[:, :db]
        return self.reshaped_np_matrix(iso)

    cpdef HilbertArray random_density(self):
        """
        Returns a random density matrix.

        >>> from qitensor import qubit, qudit

        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> ha.random_density().space
        |a><a|
        >>> ha.H.random_density().space
        |a><a|
        >>> ha.O.random_density().space
        |a><a|
        >>> (ha*hb).random_density().space
        |a,b><a,b|
        >>> (ha*hb.H).random_density()
        Traceback (most recent call last):
            ...
        HilbertError: 'not a symmetric operator space: |a><b|'
        >>> tr = (ha*hb).random_density().trace()
        >>> np.abs(tr - 1) < 1e-14
        True
        """

        if len(self.ket_set) == 0:
            return self.H.random_density()
        if len(self.bra_set) > 0 and self.bra_set != self.H.bra_set:
            raise HilbertError('not a symmetric operator space: '+str(self))

        ket_spc = self.ket_space()
        eig = np.random.rand(ket_spc.dim())
        eig /= np.sum(eig)
        U = ket_spc.random_unitary()
        return U * ket_spc.diag(eig) * U.H

    cpdef HilbertArray eye(self):
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

        bra_size = self.assert_square()

        return self.array(self.base_field.eye(bra_size), reshape=True)

    cpdef HilbertArray fully_mixed(self):
        """
        Returns the fully mixed state.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')

        >>> ha.fully_mixed()
        HilbertArray(|a><a|,
        array([[ 0.5+0.j,  0.0+0.j],
               [ 0.0+0.j,  0.5+0.j]]))

        >>> ha.fully_mixed() == ha.H.fully_mixed()
        True
        >>> ha.fully_mixed() == ha.O.fully_mixed()
        True
        """

        if len(self.ket_set) == 0 or len(self.bra_set) == 0:
            return (self * self.H).fully_mixed()

        if self.bra_set != self.H.bra_set:
            raise HilbertError('not a symmetric operator space: '+str(self))

        return self.eye() / self.ket_space().dim()

    cpdef basis_vec(self, idx):
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
            raise HilbertError('not enough indices given')
        ret[idx] = 1
        return ret

    cpdef basis(self):
        """
        Returns an orthonormal basis (the computational basis) for this space.

        >>> from qitensor import qubit, qudit, indexed_space
        >>> import numpy
        >>> ha = qubit('a')
        >>> hb = qudit('b', 5)
        >>> hc = indexed_space('c', ['x', 'y', 'z'])

        >>> spc = ha*hb*hc.H
        >>> b = spc.basis()
        >>> w = [[(x.H*y).trace() for y in b] for x in b]
        >>> numpy.allclose(w, numpy.eye(spc.dim()))
        True
        """

        return [ self.basis_vec(idx) for idx in self.index_iter() ]

    cpdef hermitian_basis(self, normalize=False, tracefree=False):
        """
        Returns an orthogonal basis (optionally normalized) of Hermitian
        operators.  It is required that the dimension of the bra space be equal
        to that of the ket space.  Real linear combinations of these basis
        operators will be Hermitian.  If the ``tracefree`` option is specified
        then the returned basis only covers the :math:`D^2-1` dimensional
        subspace.

        >>> from qitensor import qubit, qudit, indexed_space
        >>> import numpy
        >>> import numpy.random

        >>> ha = qudit('a', 7)
        >>> spc = ha.O

        >>> b = spc.hermitian_basis(normalize=True)
        >>> len(b) == ha.dim() ** 2
        True
        >>> numpy.allclose([[(x.H*y).trace() for y in b] for x in b], numpy.eye(spc.dim()))
        True
        >>> numpy.all([(x-x.H).norm() < 1e-12 for x in b])
        True
        >>> y = numpy.sum([x * numpy.random.rand() for x in b])
        >>> (y - y.H).norm() < 1e-12
        True

        >>> tf = spc.hermitian_basis(normalize=True, tracefree=True)
        >>> len(tf) == ha.dim() ** 2 - 1
        True
        >>> numpy.allclose([[(x.H*y).trace() for y in tf] for x in tf], numpy.eye(spc.dim()-1))
        True
        >>> numpy.all([(x-x.H).norm() < 1e-12 for x in tf])
        True
        >>> y = numpy.sum([x * numpy.random.rand() for x in tf])
        >>> (y - y.H).norm() < 1e-12
        True

        >>> hb = indexed_space('b', ['x', 'y', 'z'])
        >>> hc = qudit('c', 3)
        >>> spc = hb * hc.H
        >>> b = spc.hermitian_basis(normalize=True)
        >>> numpy.allclose([[(x.H*y).trace() for y in b] for x in b], numpy.eye(spc.dim()))
        True
        """

        dim = self.assert_square()
        bra_indices = list(self.bra_space().index_iter())
        ket_indices = list(self.ket_space().index_iter())
        assert dim == len(bra_indices) == len(ket_indices)

        basis = []

        if tracefree:
            c = self.base_field.frac(1, (self.base_field.sqrt(dim) + 1))
            b = 1 + (dim-2)*c
            for k in range(1, dim):
                v = self.eye() * c
                v[ ket_indices[0] + bra_indices[0] ] = 1
                v[ ket_indices[k] + bra_indices[k] ] = -b
                basis.append(v)
            #diagspc = TensorSubspace.from_span(np.eye(dim)) - TensorSubspace.from_span([np.ones(dim)])
            #for diag in diagspc:
            #    basis.append(self.diag(diag))

        for i in range(dim):
            for j in range(i, dim):
                if tracefree and i==j:
                    continue
                v = self.array()
                v[ ket_indices[i] + bra_indices[j] ] = 1
                v[ ket_indices[j] + bra_indices[i] ] = 1
                basis.append(v)

        for i in range(dim):
            for j in range(i+1, dim):
                v = self.array()
                v[ ket_indices[i] + bra_indices[j] ] = 1j
                v[ ket_indices[j] + bra_indices[i] ] = -1j
                basis.append(v)

        if normalize:
            for x in basis:
                x.normalize()

        return basis

    cpdef HilbertArray fourier_basis_state(self, int k):
        """
        Returns a state from the Fourier basis.

        The returned state is :math:`\sum_j (1/\sqrt{D}) e^{2 \pi i j k/D} |j>`
        where `D` is the dimension of the space and `j` is an integer
        (regardless of the actual values of the index set).

        >>> from qitensor import qubit, qudit, indexed_space
        >>> import numpy
        >>> numpy.set_printoptions(suppress = True)

        >>> ha = qudit('a', 4)
        >>> ha.fourier_basis_state(0)
        HilbertArray(|a>,
        array([ 0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j]))
        >>> ha.fourier_basis_state(1)
        HilbertArray(|a>,
        array([ 0.5+0.j ,  0.0+0.5j, -0.5+0.j , -0.0-0.5j]))
        >>> hb = qubit('b')
        >>> (ha*hb).fourier_basis_state(0)
        HilbertArray(|a,b>,
        array([[ 0.353553+0.j,  0.353553+0.j],
               [ 0.353553+0.j,  0.353553+0.j],
               [ 0.353553+0.j,  0.353553+0.j],
               [ 0.353553+0.j,  0.353553+0.j]]))
        >>> (ha*hb).fourier_basis_state(3) == (ha*hb).H.fourier_basis_state(3).H
        True
        >>> (ha*hb.H).fourier_basis_state(0)
        Traceback (most recent call last):
            ...
        HilbertError: 'fourier_basis_state not allowed for operators (only for bras or kets)'
        >>> hc = indexed_space('c', ['w', 'x', 'y', 'z'])
        >>> hc.fourier_basis_state(0)
        HilbertArray(|c>,
        array([ 0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j]))
        """

        if len(self.ket_set) == 0:
            return self.H.fourier_basis_state(k).H
        elif len(self.bra_set):
            raise HilbertError("fourier_basis_state not allowed for operators (only for bras or kets)")
        # else it is a ket space

        cdef int N = self.dim()
        cdef int i
        cdef np.ndarray arr = np.array([
            self.base_field.fractional_phase(i*k, N)
            for i in range(N)], dtype=self.base_field.dtype)
        arr /= self.base_field.sqrt(N)
        return self.array(data=arr, reshape=True)

    cpdef HilbertArray fourier(self):
        """
        Returns the Fourier transform gate.
        The returned operator is :math:`\sum_{jk} (1/\sqrt{D}) e^{-2 \pi i j k/D} |j><k|`

        If the bra space or ket space is empty, then the nonempty of those two
        is used to form an operator space (i.e. ``self.O``).  If both the
        bra and ket spaces are nonempty, they must be of the same dimension
        since the operator must be square.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')

        >>> ha.fourier()
        HilbertArray(|a><a|,
        array([[ 0.707107+0.j,  0.707107+0.j],
               [ 0.707107+0.j, -0.707107-0.j]]))

        >>> (ha*hb.H).fourier()
        HilbertArray(|a><b|,
        array([[ 0.707107+0.j,  0.707107+0.j],
               [ 0.707107+0.j, -0.707107-0.j]]))
        
        >>> (ha*hb).fourier()
        HilbertArray(|a,b><a,b|,
        array([[[[ 0.5+0.j ,  0.5+0.j ],
                 [ 0.5+0.j ,  0.5+0.j ]],
        <BLANKLINE>
                [[ 0.5+0.j ,  0.0-0.5j],
                 [-0.5-0.j , -0.0+0.5j]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 0.5+0.j , -0.5-0.j ],
                 [ 0.5+0.j , -0.5-0.j ]],
        <BLANKLINE>
                [[ 0.5+0.j , -0.0+0.5j],
                 [-0.5-0.j ,  0.0-0.5j]]]]))
        """

        if len(self.ket_set) == 0 or len(self.bra_set) == 0:
            return (self * self.H).fourier()

        cdef int N = self.assert_square()

        cdef np.ndarray arr = np.zeros((N, N), dtype=self.base_field.dtype)
        cdef int j, k
        for k in range(N):
            arr[1, k] = self.base_field.fractional_phase(-k, N)
        for j in range(N):
            if j == 1: continue
            for k in range(N):
                arr[j, k] = arr[1, (j*k)%N]

        arr /= self.base_field.sqrt(N)

        return self.array(data=arr, reshape=True)

    cpdef HilbertArray hadamard(self):
        """
        Returns the Hadamard matrix.
        Only applies if the dimension of the space is a power of 2.
        The returned operator is
        :math:`\sum_{jk} (1/\sqrt{D}) (-1)^{j \cdot k} |j><k|`
        where j, k are bitstrings.

        >>> from qitensor import qubit, qudit
        >>> import numpy as np
        >>> import numpy.linalg as linalg

        >>> ha = qubit('a')
        >>> ha.hadamard()
        HilbertArray(|a><a|,
        array([[ 0.707107+0.j,  0.707107+0.j],
               [ 0.707107+0.j, -0.707107+0.j]]))
        >>> ha.hadamard() == ha.O.hadamard()
        True

        >>> hb = [qubit('b%d' % i) for i in range(5)]
        >>> U = np.product([ h.hadamard() for h in hb ])
        >>> hc = qudit('c', 2**5)
        >>> V = hc.hadamard()
        >>> linalg.norm( U.as_np_matrix() - V.as_np_matrix() ) < 1e-14
        True
        """

        if len(self.ket_set) == 0 or len(self.bra_set) == 0:
            return (self * self.H).hadamard()

        cdef int N = self.assert_square()
        if _int_log2(N) < 0:
            raise HilbertError("Hadamard matrix only defined if dimension is a power of 2")

        cdef np.ndarray arr = np.zeros((N, N), dtype=self.base_field.dtype)
        cdef int j, k
        for j in range(N):
            for k in range(N):
                arr[j, k] = -1 if (1 & _countbits(j&k)) else 1

        arr /= self.base_field.sqrt(N)

        return self.array(data=arr, reshape=True)

    cpdef HilbertArray haar_matrix(self):
        """
        Returns the unitary matrix for the Haar wavelet transform.  Only
        applies if the dimension of the space is a power of 2.

        >>> from qitensor import qudit

        >>> ha = qudit('a', 1)
        >>> ha.haar_matrix()
        HilbertArray(|a><a|,
        array([[ 1.+0.j]]))

        >>> ha = qudit('a', 2)
        >>> ha.haar_matrix()
        HilbertArray(|a><a|,
        array([[ 0.707107+0.j,  0.707107+0.j],
               [ 0.707107+0.j, -0.707107+0.j]]))

        >>> ha = qudit('a', 4)
        >>> ha.haar_matrix()
        HilbertArray(|a><a|,
        array([[ 0.500000+0.j,  0.500000+0.j,  0.500000+0.j,  0.500000+0.j],
               [ 0.500000+0.j,  0.500000+0.j, -0.500000+0.j, -0.500000+0.j],
               [ 0.707107+0.j, -0.707107+0.j,  0.000000+0.j,  0.000000+0.j],
               [ 0.000000+0.j,  0.000000+0.j,  0.707107+0.j, -0.707107+0.j]]))

        >>> ha.haar_matrix() == ha.O.haar_matrix()
        True
        """

        if len(self.ket_set) == 0 or len(self.bra_set) == 0:
            return (self * self.H).haar_matrix()

        cdef int N = self.assert_square()
        cdef int n = _int_log2(N)
        if n < 0:
            raise HilbertShapeError("Hadamard matrix only defined if dimension is a power of 2")

        cdef np.ndarray arr = np.zeros((N, N), dtype=self.base_field.dtype)

        arr[0,:] = 1 / self.base_field.sqrt(N)
        cdef int row, col, i, j, k
        row = 1
        for i in range(1, n+1):
            step = 1<<(n-i)
            v = 1 / self.base_field.sqrt(step << 1)
            col = 0
            for j in range(1<<(i-1)):
                for k in range(step):
                    arr[row, col+k] = v
                    arr[row, col+k+step] = -v
                row += 1
                col += step+step

        return self.array(data=arr, reshape=True)

    cpdef full_space(self):
        """
        Returns a TensorSubspace corresponding to the entire space.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.full_space()
        <TensorSubspace of dim 2 over space (|a>)>
        >>> ha.O.full_space()
        <TensorSubspace of dim 4 over space (|a><a|)>
        """

        return TensorSubspace.full(self.shape, hilb_space=self)

    cpdef empty_space(self):
        """
        Returns a TensorSubspace corresponding to the empty space.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.empty_space()
        <TensorSubspace of dim 0 over space (|a>)>
        >>> ha.O.empty_space()
        <TensorSubspace of dim 0 over space (|a><a|)>
        """

        return TensorSubspace.empty(self.shape, hilb_space=self)

    cpdef int dim(self):
        """
        Returns the dimension of this space.

        >>> from qitensor import qubit, qudit, indexed_space
        >>> ha = qubit('a')
        >>> hb = qudit('b', 5)
        >>> hc = indexed_space('c', ['x', 'y', 'z'])
        
        >>> ha.dim()
        2
        >>> (ha*hb*hc.H).dim()
        30
        """

        return self._dim

    cpdef index_iter(self):
        """
        Returns an iterator over the indices of this space.  Each returned
        value is a tuple.

        See also: :func:`indices`, :func:`index_iter_dict`

        >>> from qitensor import qubit, qudit, indexed_space
        >>> ha = qubit('a')
        >>> hb = qudit('b', 5)
        >>> hc = indexed_space('c', ['x', 'y', 'z'])

        >>> list(ha.index_iter())
        [(0,), (1,)]

        >>> list((ha*hc).index_iter())
        [(0, 'x'), (0, 'y'), (0, 'z'), (1, 'x'), (1, 'y'), (1, 'z')]

        >>> len(list( (ha*hb*hc).index_iter() )) == (ha*hb*hc).dim()
        True

        >>> x = (ha * hb * hc.H).random_array()
        >>> norm2 = sum(abs(x[idx])**2 for idx in x.space.index_iter())
        >>> abs(norm2 - x.norm()**2) < 1e-12
        True
        """

        return itertools.product(*[s.indices for s in self.axes])

    def index_iter_dict(self):
        """
        Returns an iterator over the indices of a space.  Each returned value
        is a dictionary.

        See also: :func:`indices`, :func:`index_iter`

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')

        >>> list((ha*hb.H).index_iter_dict()) == [{ha: 0, hb.H: 0}, {ha: 0, hb.H: 1}, {ha: 1, hb.H: 0}, {ha: 1, hb.H: 1}]
        True

        >>> x = (ha*hb.H).random_unitary()
        >>> x.space
        |a><b|
        >>> [ x[idx].space for idx in ha.index_iter_dict() ]
        [<b|, <b|]
        >>> [ "%.3f" % x[idx].norm() for idx in ha.index_iter_dict() ]
        ['1.000', '1.000']
        """

        return ( dict(zip(self.axes, idx)) for idx in self.index_iter() )

    cpdef assert_ket_space(self):
        """
        Throws an exception unless the bra space is empty.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.assert_ket_space()
        >>> ha.H.assert_ket_space()
        Traceback (most recent call last):
            ...
        NotKetSpaceError: '<a|'
        >>> ha.O.assert_ket_space()
        Traceback (most recent call last):
            ...
        NotKetSpaceError: '|a><a|'
        """

        if self.bra_set:
            raise NotKetSpaceError(repr(self))

    ########## stuff that only works in Sage ##########

    cpdef reshaped_sage_matrix(self, m, input_axes=None):
        """
        Just like :func:`reshaped_np_matrix` but takes a Sage Matrix as input.

        sage: from qitensor import qubit
        sage: ha = qubit('a')
        sage: m = Matrix([[1,2],[3,4]])
        sage: ha.O.reshaped_sage_matrix(m)
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  2.+0.j],
               [ 3.+0.j,  4.+0.j]]))
        """

        if not have_sage:
            raise HilbertError('This is only available under Sage')

        return self.reshaped_np_matrix( \
            self.base_field.matrix_sage_to_np(m), \
            input_axes=input_axes)

    ########## end of stuff that only works in Sage ##########
