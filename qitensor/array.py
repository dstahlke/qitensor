"""
A HilbertArray is a vector in a HilbertSpace.  Internally, it is backed by a
numpy.array.  HilbertArray's are to be created using the
:meth:`HilbertSpace.array` method.
"""

import numpy as np

from qitensor import have_sage, shape_product
from qitensor.exceptions import BraKetMixtureError, DuplicatedSpaceError, \
    HilbertError, HilbertIndexError, HilbertShapeError, HilbertSliceError, \
    NotKetSpaceError, MismatchedIndexSetError
from qitensor.space import HilbertSpace
from qitensor.atom import HilbertAtom
from qitensor.arrayformatter import FORMATTER

__all__ = ['HilbertArray']

class HilbertArray(object):
    def __init__(self, space, data, noinit_data, reshape, input_axes):
        """
        Don't call this constructor yourself, use HilbertSpace.array
        """

        hs = space
        self.space = hs
        self.axes = hs._array_axes

        if noinit_data:
            assert data is None
            assert input_axes is None
            self.nparray = None
        elif data is None:
            assert input_axes is None
            self.nparray = np.zeros(hs.shape, dtype=hs.base_field.dtype)
        else:
            self.nparray = np.array(data, dtype=hs.base_field.dtype)

            if input_axes is None:
                data_shape = hs.shape
            else:
                data_shape = tuple([ len(spc.indices) for spc in input_axes ])

            # make sure given array is the right size
            if reshape:
                if shape_product(self.nparray.shape) != shape_product(data_shape):
                    raise HilbertShapeError(shape_product(data.shape),
                        shape_product(data_shape))
                self.nparray = self.nparray.reshape(data_shape)
            if self.nparray.shape != data_shape:
                raise HilbertShapeError(self.nparray.shape, data_shape)

            if input_axes is not None:
                assert frozenset(input_axes) == frozenset(self.axes)
                shuffle = [ input_axes.index(x) for x in self.axes ]
                self.nparray = self.nparray.transpose(shuffle)

        if self.nparray is not None:
            cast_fn = space.base_field.input_cast_function()
            if cast_fn is not None:
                self.nparray = np.vectorize(cast_fn)(self.nparray)

    def copy(self):
        """
        Creates a copy (not a view) of this array.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.array([1, 2]); x
        HilbertArray(|a>,
        array([ 1.+0.j,  2.+0.j]))
        >>> y = x.copy()
        >>> y[0] = 3
        >>> y
        HilbertArray(|a>,
        array([ 3.+0.j,  2.+0.j]))
        >>> x
        HilbertArray(|a>,
        array([ 1.+0.j,  2.+0.j]))
        """

        ret = self.space.array(noinit_data=True)
        ret.nparray = self.nparray.copy()
        return ret

    def _reassign(self, other):
        """
        Used internally to change the contents of a HilbertArray without creating a new object.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_unitary()
        >>> y = ha.random_unitary()
        >>> z = x
        >>> x *= y
        >>> x is z
        True
        """

        self.space = other.space
        self.nparray = other.nparray
        self.axes = other.axes

    def get_dim(self, atom):
        """
        Returns the axis corresponding to the given HilbertAtom.

        This is useful when working with the underlying numpy array.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> x = (ha * hb).array([[1, 2], [4, 8]])
        >>> [x.get_dim(h) for h in (ha, hb)]
        [0, 1]
        >>> x.nparray.sum(axis=x.get_dim(ha))
        array([  5.+0.j,  10.+0.j])
        >>> x.nparray.sum(axis=x.get_dim(hb))
        array([  3.+0.j,  12.+0.j])
        """

        return self.space._array_axes_lookup[atom]

    def _assert_same_axes(self, other):
        """
        Throws an exception if self.axes != other.axes.  Used when determining whether
        two arrays are compatible for certain operations (such as addition).

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> x = ha.array()
        >>> y = ha.array()
        >>> z = hb.array()
        >>> x._assert_same_axes(y)
        >>> x._assert_same_axes(z)
        Traceback (most recent call last):
            ...
        HilbertIndexError: 'Mismatched HilbertSpaces: |a> vs. |b>'
        """

        if self.axes != other.axes:
            raise HilbertIndexError('Mismatched HilbertSpaces: '+
                repr(self.space)+' vs. '+repr(other.space))

    def set_data(self, new_data):
        """
        Sets this array equal to the given argument.

        :param new_data: the new data
        :type new_data: HilbertArray or anything that can be made into a
            numpy.array

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> x = (ha*hb).array()
        >>> y = (ha*hb).random_array()
        >>> x.set_data(y)
        >>> x == y
        True
        >>> x.set_data([[1, 2], [3, 4]])
        >>> x
        HilbertArray(|a,b>,
        array([[ 1.+0.j,  2.+0.j],
               [ 3.+0.j,  4.+0.j]]))
        """

        if isinstance(new_data, HilbertArray):
            self._assert_same_axes(new_data)
            self.set_data(new_data.nparray)
        else:
            # This is needed to make slices work properly
            self.nparray[:] = new_data

    def tensordot(self, other, contraction_spaces=None):
        """
        Inner or outer product of two arrays.

        :param other: the other array taking place in this operation
        :type other: HilbertArray
        :param contraction_spaces: the spaces on which to do a tensor
            contraction
        :type other: None, frozenset, or HilbertSpace; default None

        If ``contraction_spaces`` is ``None`` (the default), contraction will
        be across the intersection of the bra space of this array and the ket
        space of ``other``.  If a ``frozenset`` is given, it should consist of
        ``HilbertAtom`` objects which are kets.  If a ``HilbertSpace`` is
        given, it must be a ket space.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> x = (ha * hb.H * hc.H).random_array()
        >>> x.space
        |a><b,c|
        >>> y = (hc * ha.H).random_array()
        >>> y.space
        |c><a|
        >>> x.tensordot(y) == x * y
        True
        >>> x.tensordot(y).space
        |a><a,b|
        >>> x.tensordot(y, frozenset()).space
        |a,c><a,b,c|
        >>> x.tensordot(y, hc).space
        |a><a,b|
        >>> (ha.bra(0) * hb.bra(0)) * (ha.ket(0) * hb.ket(0))
        (1+0j)
        >>> (ha.bra(0) * hb.bra(0)) * (ha.ket(0) * hb.ket(1))
        0j
        >>> xa = ha.O.random_array()
        >>> xb = hb.O.random_array()
        >>> ((xa*xa)*(xb*xb)).space
        |a,b><a,b|
        >>> ((xa*xa)*(xb*xb)).closeto((xa*xb)*(xa*xb))
        True
        """

        hs = self.space
        ohs = other.space
        #print str(hs)+'*'+str(ohs)

        hs.base_field.assert_same(ohs.base_field)

        # shortcut for common case
        if (contraction_spaces is None) and (hs._is_simple_dyad) and (hs == ohs) and (hs == hs.H):
            return self.space.array(np.dot(self.nparray, other.nparray))

        if contraction_spaces is None:
            mul_space = frozenset([x.H for x in hs.bra_set]) & ohs.ket_set
        elif isinstance(contraction_spaces, frozenset):
            mul_space = contraction_spaces
            for x in mul_space:
                if not isinstance(x, HilbertSpace):
                    raise TypeError('contraction space must consist of '+
                        'HilbertAtoms')
                if x.is_dual():
                    raise NotKetSpaceError('contraction space must consist of '+
                        'kets')
        elif isinstance(contraction_spaces, HilbertSpace):
            if len(contraction_spaces.bra_set) > 0:
                raise NotKetSpaceError('contraction space must consist of kets')
            mul_space = contraction_spaces.ket_set
        else:
            raise TypeError('contraction space must be HilbertSpace '+
                'or frozenset')

        for x in mul_space:
            assert isinstance(x, HilbertAtom)
            assert not x.is_dual

        mul_H = frozenset([x.H for x in mul_space])
        #print mul_space

        mul_space_sorted = sorted(mul_space)
        axes_self  = [self.get_dim(x.H) for x in mul_space_sorted]
        axes_other = [other.get_dim(x)  for x in mul_space_sorted]
        #print axes_self, axes_other
        td = np.tensordot(self.nparray, other.nparray,
            axes=(axes_self, axes_other))
        assert td.dtype == hs.base_field.dtype

        #print "hs.k", hs.sorted_kets
        #print "hs.b", hs.sorted_bras
        #print "ohs.k", ohs.sorted_kets
        #print "ohs.b", ohs.sorted_bras
        #print "cH", mul_H
        #print "c", mul_space
        td_axes = \
            hs.sorted_kets + \
            [x for x in hs.sorted_bras if not x in mul_H] + \
            [x for x in ohs.sorted_kets if not x in mul_space] + \
            ohs.sorted_bras
        #print td_axes
        #print td.shape
        assert len(td_axes) == len(td.shape)

        if len(td_axes) == 0:
            # convert 0-d array to scalar
            return td[()]
        else:
            ket1 = hs.ket_set
            ket2 = (ohs.ket_set-mul_space)
            bra1 = (hs.bra_set-mul_H)
            bra2 = ohs.bra_set

            if not (ket1.isdisjoint(ket2) and bra1.isdisjoint(bra2)):
                raise DuplicatedSpaceError(
                    hs.base_field.create_space2(ket1 & ket2, bra1 & bra2))

            ret_space = hs.base_field.create_space2(ket1 | ket2, bra1 | bra2)
            #print 'ret', ret_space

            ret = ret_space.array(noinit_data=True)
            #print "ret", ret.axes
            permute = tuple([td_axes.index(x) for x in ret.axes])
            #print permute
            ret.nparray = td.transpose(permute)

            return ret

    def transpose(self, tpose_axes=None):
        """
        Perform a transpose or partial transpose operation.

        :param tpose_axes: the space on which to transpose
        :type tpose_axes: HilbertSpace or None; default None

        If ``tpose_axes`` is ``None`` a full transpose is performed.
        Otherwise, ``tpose_axes`` should be a ``HilbertSpace``.  The array will
        be transposed across all axes which are part of the bra space or ket
        space of ``tpose_axes``.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> x = ha.O.array([[1, 2], [3, 4]]); x
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  2.+0.j],
               [ 3.+0.j,  4.+0.j]]))
        >>> x.transpose()
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  3.+0.j],
               [ 2.+0.j,  4.+0.j]]))
        >>> x.transpose() == x.T
        True
        >>> y = (ha * hb).random_array()
        >>> y.space
        |a,b>
        >>> y.transpose(ha).space
        |b><a|
        >>> y.transpose(ha) == y.transpose(ha.H)
        True
        """

        if tpose_axes is None:
            tpose_axes = self.space

        tpose_atoms = []
        for x in tpose_axes.bra_set | tpose_axes.ket_set:
            if not (x in self.axes or x.H in self.axes):
                raise HilbertIndexError('Hilbert space not part of this '+
                    'array: '+repr(x))
            if x.is_dual:
                tpose_atoms.append(x.H)
            else:
                tpose_atoms.append(x)

        in_space_dualled = []
        for x in self.axes:
            y = x.H if x.is_dual else x
            if y in tpose_atoms:
                in_space_dualled.append(x.H)
            else:
                in_space_dualled.append(x)

        out_space = self.space.base_field.create_space1(in_space_dualled)

        ret = out_space.array(noinit_data=True)
        permute = tuple([in_space_dualled.index(x) for x in ret.axes])
        ret.nparray = self.nparray.transpose(permute)

        return ret

    def relabel(self, from_spaces, to_spaces=None):
        """
        Returns a HilbertArray with the same data as this one, but with axes relabelled.

        :param from_space: the old space, or list of spaces, or dict mapping old->new
        :type from_space: HilbertSpace, list, dict
        :param to_space: the new space (or list of spaces)
        :type to_space: HilbertSpace, list

        This method changes the labels of the axes of this array.  It is permitted to
        change a bra to a ket or vice-versa, in which case the effect is equivalent to
        doing a partial transpose.  There are three ways to call this method:

        * You can pass two HilbertSpaces, in which case the first space will be
          relabeled to the second one.  If the passed spaces are not HilbertAtoms
          (i.e. they are composites like ``|a,b>``) then the mapping from atom to
          atom is done using the same alphabetical sorting that is used when
          displaying the name of the space.  In this case, the other two ways of
          calling relabel would probably be clearer.
        * You can pass a pair of tuples of HilbertAtoms.  In this case the first
          space from the first list gets renamed to the first space from the second
          list, and so on.
        * You can pass a dictionary of HilbertAtoms of the form {from_atom => to_atom, ...}.

        FIXME - does it return a view? should it?

        >>> import numpy
        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> x = (ha * hb).random_array()
        >>> x.space
        |a,b>
        >>> x.relabel(hb, hc).space
        |a,c>
        >>> numpy.allclose(x.relabel(hb, hc).nparray, x.nparray)
        True
        >>> x.relabel(ha, hc).space
        |b,c>
        >>> # underlying nparray is different because axes order changed
        >>> numpy.allclose(x.relabel(ha, hc).nparray, x.nparray)
        False
        >>> x.relabel(ha * hb, ha.prime * hb.prime).space
        |a',b'>
        >>> y = ha.O.array()
        >>> y.space
        |a><a|
        >>> # relabeling is needed because |a,a> is not allowed
        >>> y.relabel(ha.H, ha.prime.H).transpose(ha.prime).space
        |a,a'>
        >>> z = (ha*hb.H*hc.O).random_array()
        >>> z.space
        |a,c><b,c|
        >>> z1 = z.relabel((ha, hc.H), (ha.H, hc.prime))
        >>> z1.space
        |c,c'><a,b|
        >>> z2 = z.relabel({ha: ha.H, hc.H: hc.prime})
        >>> z2.space
        |c,c'><a,b|
        >>> z1 == z2
        True
        >>> z == z1.relabel({ha.H: ha, hc.prime: hc.H})
        True
        """

        if isinstance(from_spaces, HilbertSpace):
            from_spaces = (from_spaces, )
        if isinstance(to_spaces, HilbertSpace):
            to_spaces = (to_spaces, )

        if isinstance(from_spaces, dict):
            assert to_spaces is None
            mapping = from_spaces
            HilbertSpace._assert_nodup_space(mapping.values(), "an output space was listed twice")
        else:
            # Cast to list, in case we were given a generator.  Throw error if not list-like.
            from_spaces = list(from_spaces)
            to_spaces = list(to_spaces)

            from_spaces = HilbertSpace._expand_list_to_atoms(from_spaces)
            to_spaces   = HilbertSpace._expand_list_to_atoms(to_spaces)

            assert len(from_spaces) == len(to_spaces)
            HilbertSpace._assert_nodup_space(from_spaces, "an input space was listed twice")
            HilbertSpace._assert_nodup_space(to_spaces, "an output space was listed twice")
            mapping = dict(zip(from_spaces, to_spaces))

        for (k,v) in mapping.iteritems():
            assert isinstance(k, HilbertAtom)
            assert isinstance(v, HilbertAtom)
            if not k in self.space.bra_ket_set:
                raise MismatchedIndexSetError("not in input space: "+repr(k))

        xlate_list = [ mapping[x] if x in mapping else x for x in self.axes ]
        HilbertSpace._assert_nodup_space(xlate_list, "relabling would cause a duplicated space")
        new_space = self.space.base_field.create_space1(xlate_list)

        return new_space.array(data=self.nparray, input_axes=xlate_list)

    def relabel_prime(self):
        """
        Returns a relabeled array with primed spaces.

        See also: :func:`relabel`, :func:`qitensor.atom.HilbertAtom.prime`

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> x = (ha*hb).array()
        >>> x.relabel_prime().space
        |a',b'>
        >>> (x * x.relabel_prime()).space
        |a,a',b,b'>
        """

        return self.relabel(self.axes, [x.prime for x in self.axes])

    def apply_map(self, fn):
        """
        Apply the given function to each element of the array.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> v = ha.array([1, -2])
        >>> v.apply_map(lambda x: abs(x))
        HilbertArray(|a>,
        array([ 1.+0.j,  2.+0.j]))
        """

        dtype = self.space.base_field.dtype
        arr = np.vectorize(fn, otypes=[dtype])(self.nparray)
        return self.space.array(arr)

    def closeto(self, other, rtol=1e-05, atol=1e-08):
        """
        Checks whether two arrays are nearly equal, similar to numpy.allclose.

        For details, see the numpy.allclose documentation.

        >>> from qitensor import qudit
        >>> ha = qudit('a', 10)
        >>> x = ha.random_array()
        >>> y = ha.random_array()
        >>> x.closeto(y)
        False
        >>> x.closeto(x * (1+1e-10))
        True
        >>> x == x * (1+1e-10)
        False
        """

        if not isinstance(other, HilbertArray):
            raise TypeError('other must be HilbertArray')
        self._assert_same_axes(other)
        return np.allclose(self.nparray, other.nparray, rtol=rtol, atol=atol)

    def __eq__(self, other):
        """
        Checks whether two arrays are equal.

        >>> from qitensor import qudit
        >>> ha = qudit('a', 10)
        >>> hb = qudit('b', 10)
        >>> x = ha.array()
        >>> y = ha.random_array()
        >>> x == x and y == y
        True
        >>> x == y
        False
        >>> x == y * 0
        True
        >>> x == x.relabel({ ha: hb })
        False
        """

        if not isinstance(other, HilbertArray):
            return False
        elif self.space != other.space:
            return False
        else:
            return np.all(self.nparray == other.nparray)

    def __ne__(self, other):
        """
        Checks whether two arrays are not equal.

        >>> from qitensor import qudit
        >>> ha = qudit('a', 10)
        >>> hb = qudit('b', 10)
        >>> x = ha.array()
        >>> y = ha.random_array()
        >>> x != x or y != y
        False
        >>> x != y
        True
        >>> x != y * 0
        False
        >>> x != x.relabel({ ha: hb })
        True
        """

        return not (self == other)

    def lmul(self, other):
        """
        Returns other*self.

        This is useful for listing operations in chronoligical order when
        implementing quantum circuits.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> state = ha.random_array()
        >>> ha.X * ha.Y * state == state.lmul(ha.Y).lmul(ha.X)
        True
        """

        return other * self

    def __mul__(self, other):
        """
        Multiplies two arrays, or an array and a scalar.
        
        Given two arrays, contracts between the common bra spaces of the left
        argument and ket spaces of the right argument.  This is equivalent to
        calling self.tensordot(other).

        Given an array and scalar, does the obvious thing.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> hx = qubit('x')
        >>> hy = qubit('y')
        >>> q = (ha*hb.H*hc.H).random_array()
        >>> q.space
        |a><b,c|
        >>> r = (hc*hx*hy.H).random_array()
        >>> r.space
        |c,x><y|
        >>> (q*r).space
        |a,x><b,y|
        >>> (q+q).closeto(2*q)
        True
        >>> (q+q).closeto(q*2)
        True
        >>> q+r # FIXME - in __add__
        Traceback (most recent call last):
            ...
        HilbertIndexError: 'Mismatched HilbertSpaces: |a><b,c| vs. |c,x><y|'
        """

        if isinstance(other, HilbertArray):
            return self.tensordot(other)
        else:
            ret = self.copy()
            ret *= other
            return ret

    def __imul__(self, other):
        """
        In-place multiplication.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_unitary()
        >>> y = ha.random_unitary()
        >>> x_copy = x.copy()
        >>> x_ref  = x
        >>> x *= y
        >>> x is x_ref
        True
        >>> x is x_copy
        False
        >>> x *= 2
        >>> x is x_ref
        True
        >>> x.closeto(x_copy*y*2)
        True
        """

        if isinstance(other, HilbertArray):
            self._reassign(self * other)
        else:
            # hopefully other is a scalar
            self.nparray *= other
        return self

    def __rmul__(self, other):
        """
        Reflected multiplication.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_array()
        >>> (x*2).closeto(2*x)
        True
        """

        return self * other

    def __add__(self, other):
        """
        Adds two arrays.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> x = ha.random_array()
        >>> y = hb.random_array()
        >>> (x+x).closeto(2*x)
        True
        >>> x+y
        Traceback (most recent call last):
            ...
        HilbertIndexError: 'Mismatched HilbertSpaces: |a> vs. |b>'
        """

        if not isinstance(other, HilbertArray):
            raise TypeError('HilbertArray can only add another HilbertArray')
        ret = self.copy()
        ret += other
        return ret

    def __iadd__(self, other):
        """
        In-place addition.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_array()
        >>> y = ha.random_array()
        >>> x_copy = x.copy()
        >>> x_ref  = x
        >>> x += y
        >>> x is x_ref
        True
        >>> x is x_copy
        False
        >>> x.closeto(x_copy+y)
        True
        """

        if not isinstance(other, HilbertArray):
            raise TypeError('HilbertArray can only add another HilbertArray')
        self._assert_same_axes(other)
        self.nparray += other.nparray
        return self

    def __sub__(self, other):
        """
        Subtracts two arrays.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> x = ha.random_array()
        >>> y = hb.random_array()
        >>> (2*x-x).closeto(x)
        True
        >>> x-y
        Traceback (most recent call last):
            ...
        HilbertIndexError: 'Mismatched HilbertSpaces: |a> vs. |b>'
        """

        if not isinstance(other, HilbertArray):
            raise TypeError('HilbertArray can only subtract '+
                'another HilbertArray')
        ret = self.copy()
        ret -= other
        return ret

    def __neg__(self):
        """
        Returns negation of this array.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_array()
        >>> y = -x
        >>> (x+y).norm() == 0
        True
        """

        return self * -1

    def __isub__(self, other):
        """
        In-place subtraction.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_array()
        >>> y = ha.random_array()
        >>> x_copy = x.copy()
        >>> x_ref  = x
        >>> x -= y
        >>> x is x_ref
        True
        >>> x is x_copy
        False
        >>> x.closeto(x_copy-y)
        True
        """

        if not isinstance(other, HilbertArray):
            raise TypeError('HilbertArray can only subtract '+
                'another HilbertArray')
        self._assert_same_axes(other)
        self.nparray -= other.nparray
        return self

    def __div__(self, other):
        """
        Divide by a scalar.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_array()
        >>> (x/2).closeto(x*0.5)
        True
        >>> # the following exception is thrown from within numpy
        >>> x / x
        Traceback (most recent call last):
            ...
        TypeError: unsupported operand type(s) for /: 'complex' and 'HilbertArray'
        """

        ret = self.copy()
        ret /= other
        return ret

    def __idiv__(self, other):
        """
        In-place division by a scalar.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_array()
        >>> x_copy = x.copy()
        >>> x_ref  = x
        >>> x /= 2
        >>> x is x_ref
        True
        >>> x is x_copy
        False
        >>> x.closeto(x_copy*0.5)
        True
        """

        self.nparray /= other
        return self

    def __truediv__(self, other):
        """
        Divide by a scalar.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_array()
        >>> (x/2).closeto(x*0.5)
        True
        >>> # the following exception is thrown from within numpy
        >>> x / x
        Traceback (most recent call last):
            ...
        TypeError: unsupported operand type(s) for /: 'complex' and 'HilbertArray'
        """

        ret = self.copy()
        ret.__itruediv__(other)
        return ret

    def __itruediv__(self, other):
        """
        In-place division by a scalar.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_array()
        >>> x_copy = x.copy()
        >>> x_ref  = x
        >>> x /= 2
        >>> x is x_ref
        True
        >>> x is x_copy
        False
        >>> x.closeto(x_copy*0.5)
        True
        """

        self.nparray.__itruediv__(other)
        return self

    def __pow__(self, other):
        if self.space != self.space.H:
            raise HilbertError('bra space must be the same as ket space '+
                '(space was '+repr(self.space)+')')
        return self.np_matrix_transform( \
            lambda x: self.space.base_field.mat_pow(x, other))

    def __ipow__(self, other):
        self.nparray[:] = self.__pow__(other).nparray
        return self

    def _index_key_to_map(self, key):
        index_map = {}
        if isinstance(key, dict):
            for (k, v) in key.iteritems():
                if not isinstance(k, HilbertSpace):
                    raise TypeError('not a HilbertSpace: '+repr(k))
                if not isinstance(v, list) and not isinstance(v, tuple):
                    v = (v,)
                atoms = k.sorted_kets + k.sorted_bras
                if len(atoms) != len(v):
                    raise HilbertIndexError("Number of indices doesn't match "+
                        "number of HilbertSpaces")
                for (hilb, idx) in zip(atoms, v):
                    index_map[hilb] = idx
        elif isinstance(key, tuple) or isinstance(key, list):
            if len(key) != len(self.axes):
                raise HilbertIndexError("Wrong number of indices given "+
                    "(%d for %s)" % (len(key), str(self.space)))
            for (i, k) in enumerate(key):
                if isinstance(k, slice):
                    if k == slice(None):
                        # full slice is the same as not specifying anything for
                        # this axis
                        pass
                    else:
                        raise HilbertSliceError("Slices are not allowed")
                else:
                    index_map[self.axes[i]] = k
        else:
            if len(self.axes) != 1:
                raise HilbertIndexError("Wrong number of indices given "+
                    "(1 for %s)" % str(self.space))
            if isinstance(key, slice):
                if key == slice(None):
                    # full slice is the same as not specifying anything for
                    # this axis
                    pass
                else:
                    raise HilbertSliceError("Slices are not allowed")
            else:
                index_map[self.axes[0]] = key

        for (spc, idx) in index_map.iteritems():
            if not isinstance(spc, HilbertSpace):
                raise TypeError('not a HilbertSpace: '+repr(spc))
            if not spc in self.axes:
                raise HilbertIndexError('Hilbert space not part of this '+
                    'array: '+repr(spc))

        return index_map

    def _get_set_item(self, key, do_set=False, set_val=None):
        index_map = self._index_key_to_map(key)

        out_axes = []
        slice_list = []
        for x in self.axes:
            if x in index_map:
                try:
                    idx_n = x.indices.index(index_map[x])
                except ValueError:
                    raise HilbertIndexError('Index set for '+repr(x)+' does '+
                        'not contain '+repr(index_map[x]))
                slice_list.append(idx_n)
            else:
                slice_list.append(slice(None))
                out_axes.append(x)
        assert len(slice_list) == len(self.nparray.shape)

        if do_set and len(out_axes) == 0:
            # must do assignment like this, since in the 1-d case numpy will
            # return a scalar rather than a view
            self.nparray[tuple(slice_list)] = set_val

        sliced = self.nparray[tuple(slice_list)]

        if len(out_axes) == 0:
            # Return a scalar, not a HilbertArray.
            # We already did do_set, if applicable.
            return sliced
        else:
            assert len(sliced.shape) == len(out_axes)
            ret = self.space.base_field.create_space1(out_axes). \
                array(noinit_data=True)
            permute = tuple([out_axes.index(x) for x in ret.axes])
            ret.nparray = sliced.transpose(permute)

            if do_set:
                ret.set_data(set_val)

            return ret

    def __getitem__(self, key):
        return self._get_set_item(key)

    def __setitem__(self, key, val):
        return self._get_set_item(key, True, val)

    def _get_row_col_spaces(self, row_space=None, col_space=None):
        """
        Parses the row_space and col_space parameters used by various functions.
        """

        def parse_space(s):
            if s is None:
                return None
            elif isinstance(s, HilbertSpace):
                return s.sorted_kets + s.sorted_bras
            else:
                return sum((parse_space(x) for x in s), [])

        col_space = parse_space(col_space)
        row_space = parse_space(row_space)

        if row_space is None and col_space is None:
            col_space = self.space.sorted_kets
            row_space = self.space.sorted_bras
        elif row_space is None:
            row_space = [x for x in self.axes if not x in col_space]
        elif col_space is None:
            col_space = [x for x in self.axes if not x in row_space]

        col_set = frozenset(col_space)
        row_set = frozenset(row_space)
        assert col_set.isdisjoint(row_set)
        assert row_set <= self.space.bra_ket_set
        assert col_set <= self.space.bra_ket_set
        assert col_set | row_set == self.space.bra_ket_set

        return (row_space, col_space)

    def as_np_matrix(self, dtype=None, row_space=None, col_space=None):
        # FIXME - docs for row_space and col_space params
        """
        Returns the underlying data as a numpy.matrix.  Returns a copy, not a view.

        >>> import numpy
        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> x = (ha.O * hb).random_array()
        >>> x.space
        |a,b><a|
        >>> x.as_np_matrix().shape
        (6, 2)
        >>> # returns a copy, not a view
        >>> x.as_np_matrix().fill(0); x.norm() == 0
        False
        >>> x.as_np_matrix(col_space=ha.O).shape
        (4, 3)
        >>> x.as_np_matrix(row_space=ha.O).shape
        (3, 4)
        >>> numpy.allclose(x.as_np_matrix(col_space=ha.O), x.as_np_matrix(row_space=ha.O).T)
        True
        """

        # shortcut for common case
        if self.space._is_simple_dyad:
            return np.matrix(self.nparray, dtype=dtype)

        rowcol_kw = { 'row_space': row_space, 'col_space': col_space }
        (row_space, col_space) = self._get_row_col_spaces(**rowcol_kw)

        col_size = shape_product([x.dim() for x in col_space])
        row_size = shape_product([x.dim() for x in row_space])
        axes = [self.get_dim(x) for x in col_space + row_space]

        #print col_size, row_size, axes

        v = self.nparray.transpose(axes).reshape(col_size, row_size)
        return np.matrix(v, dtype=dtype)

    def np_matrix_transform(self, f, transpose_dims=False):
        """
        Performs a matrix operation.

        :param f: operation to perform
        :type f: lambda function
        :param transpose_dims: if True, the resultant Hilbert space is
            transposed
        :type transpose_dims: bool

        >>> from qitensor import qubit
        >>> import numpy.linalg
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> x = (ha * hb.H).random_array()
        >>> x.space
        |a><b|
        >>> y = x.np_matrix_transform(numpy.linalg.inv, transpose_dims=True)
        >>> y.space
        |b><a|
        >>> y == x.I
        True
        """

        m = self.as_np_matrix()
        m = f(m)
        out_hilb = self.space
        if transpose_dims:
            out_hilb = out_hilb.H
        return out_hilb.reshaped_np_matrix(m)

    @property
    def H(self):
        """
        Returns the adjoint (Hermitian conjugate) of this array.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.array([1j, 0]); x
        HilbertArray(|a>,
        array([ 0.+1.j,  0.+0.j]))
        >>> x.H
        HilbertArray(<a|,
        array([ 0.-1.j,  0.-0.j]))
        >>> y = ha.O.array([[1+2j, 3+4j], [5+6j, 7+8j]]); y
        HilbertArray(|a><a|,
        array([[ 1.+2.j,  3.+4.j],
               [ 5.+6.j,  7.+8.j]]))
        >>> y.H
        HilbertArray(|a><a|,
        array([[ 1.-2.j,  5.-6.j],
               [ 3.-4.j,  7.-8.j]]))
        """

        return self.np_matrix_transform( \
            self.space.base_field.mat_adjoint, \
            transpose_dims=True)

    @property
    def I(self):
        """
        Returns the matrix inverse of this array.

        It is required that the dimension of the bra space be equal to the
        dimension of the ket space.

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> x = ha.O.random_array()
        >>> (x * x.I - ha.eye()).norm() < 1e-13
        True
        >>> hb = qubit('b')
        >>> hc = qudit('c', 4)
        >>> y = (ha * hb * hc.H).random_array()
        >>> (y * y.I - (ha * hb).eye()).norm() < 1e-13
        True
        >>> (y.I * y - hc.eye()).norm() < 1e-13
        True
        """

        return self.np_matrix_transform( \
            self.space.base_field.mat_inverse, \
            transpose_dims=True)

    @property
    def T(self):
        """
        Returns the transpose of this array.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.array([1j, 0]); x
        HilbertArray(|a>,
        array([ 0.+1.j,  0.+0.j]))
        >>> x.T
        HilbertArray(<a|,
        array([ 0.+1.j,  0.+0.j]))
        >>> y = ha.O.array([[1+2j, 3+4j], [5+6j, 7+8j]]); y
        HilbertArray(|a><a|,
        array([[ 1.+2.j,  3.+4.j],
               [ 5.+6.j,  7.+8.j]]))
        >>> y.T
        HilbertArray(|a><a|,
        array([[ 1.+2.j,  5.+6.j],
               [ 3.+4.j,  7.+8.j]]))
        """

        # transpose should be the same for all base_field's
        return self.np_matrix_transform(lambda x: x.T, transpose_dims=True)

    @property
    def O(self):
        """
        Makes a density operator from a pure state.

        The input must be a ket vector.  The output is ``self * self.H``.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.array([1j, 2]); x
        HilbertArray(|a>,
        array([ 0.+1.j,  2.+0.j]))
        >>> x.O
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  0.+2.j],
               [ 0.-2.j,  4.+0.j]]))
        """

        if self.space.bra_set:
            raise NotKetSpaceError('self.O only applies to ket spaces')
        else:
            return self * self.H

    def det(self):
        """
        Returns the matrix determinant of this array.

        It is required that the dimension of the bra space be equal to the
        dimension of the ket space.

        >>> import numpy.linalg
        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qudit('c', 4)
        >>> y = (ha * hb * hc.H).random_array()
        >>> abs( y.det() - numpy.linalg.det(y.as_np_matrix()) ) < 1e-14
        True
        """

        return self.space.base_field.mat_det(self.as_np_matrix())

    def fill(self, val):
        """
        Fills every entry of this array with a constant value.

        NOTE: the array is modified in-place and is not returned.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.random_array()
        >>> x.fill(2)
        >>> x
        HilbertArray(|a>,
        array([ 2.+0.j,  2.+0.j]))
        """

        # fill should be the same for all base_field's
        self.nparray.fill(val)

    def norm(self):
        """
        Returns the norm of this array.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.array([3, 4])
        >>> x.norm()
        5.0
        >>> y = ha.O.array([[1, 2], [3, 4]])
        >>> y.norm() ** 2
        30.0
        """

        return self.space.base_field.mat_norm(self.nparray)

    def normalize(self):
        """
        Normalizes array in-place.

        See also: :func:`normalized`

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.array([3, 4])
        >>> x.normalize()
        >>> x
        HilbertArray(|a>,
        array([ 0.6+0.j,  0.8+0.j]))
        """

        self /= self.norm()

    def normalized(self):
        """
        Returns a normalized copy of this array.

        See also: :func:`normalize`

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.array([3, 4])
        >>> x.normalized()
        HilbertArray(|a>,
        array([ 0.6+0.j,  0.8+0.j]))
        """

        return self / self.norm()

    def pinv(self, rcond=1e-15):
        """
        Returns the Moore-Penrose pseudoinverse of this array.

        :param rcond: cutoff for small singular values (see numpy.linalg.pinv
            docs for more info)
        :type rcond: float; default 1e-15

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> x = (ha * hb.H).random_array()
        >>> x.as_np_matrix().shape
        (2, 3)
        >>> (x * x.pinv() - ha.eye()).norm() < 1e-13
        True
        """

        return self.np_matrix_transform( \
            lambda x: self.space.base_field.mat_pinv(x, rcond), \
            transpose_dims=True)

    def conj(self):
        """
        Returns the complex conjugate of this array.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> x = ha.array([1j, 0]); x
        HilbertArray(|a>,
        array([ 0.+1.j,  0.+0.j]))
        >>> x.conj()
        HilbertArray(|a>,
        array([ 0.-1.j,  0.-0.j]))
        >>> y = ha.O.array([[1+2j, 3+4j], [5+6j, 7+8j]]); y
        HilbertArray(|a><a|,
        array([[ 1.+2.j,  3.+4.j],
               [ 5.+6.j,  7.+8.j]]))
        >>> y.conj()
        HilbertArray(|a><a|,
        array([[ 1.-2.j,  3.-4.j],
               [ 5.-6.j,  7.-8.j]]))
        """

        return self.np_matrix_transform( \
            self.space.base_field.mat_conj)

    def trace(self, axes=None):
        """
        Returns the (full or partial) trace of this array.

        FIXME - update docs with advanced trace features

        :param axes: axes to trace over, all axes if None (in which case the bra
            space must be the same as the ket space)
        :type axes: HilbertSpace; default None

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> hc = qubit('c')
        >>> x = ha.O.random_array()
        >>> y = hb.O.random_array()
        >>> z = hc.random_array()
        >>> abs(x.trace() - (x[0, 0] + x[1, 1])) < 1e-14
        True
        >>> abs(y.trace() - (y[0, 0] + y[1, 1] + y[2, 2])) < 1e-14
        True
        >>> abs(x.trace() * y.trace() - (x*y).trace()) < 1e-14
        True
        >>> n = hb.random_array().normalized()
        >>> # trace of a projector
        >>> abs( (n*n.H).trace() - 1 ) < 1e-14
        True
        >>> ( (x*y).trace(ha) - x.trace() * y ).norm() < 1e-14
        True
        >>> ( (x*y).trace(hb) - x * y.trace() ).norm() < 1e-14
        True
        >>> abs( (x*y).trace(ha*hb) - (x*y).trace() ) < 1e-14
        True
        >>> abs( (x*y).trace(ha).trace(hb) - (x*y).trace() ) < 1e-14
        True
        >>> abs( x.trace(ha) - x.trace(ha.H) ) < 1e-14
        True
        >>> abs( x.trace(ha) - x.trace(ha.O) ) < 1e-14
        True
        >>> ( (x*z).trace(ha) - x.trace() * z ).norm() < 1e-14
        True
        >>> ( (x*z.H).trace(ha) - x.trace() * z.H ).norm() < 1e-14
        True

        >>> # trace between a pair of ket spaces (map-state duality stuff)
        >>> n = hb.random_array().normalized()
        >>> w = n.H.relabel({ hb.H: hb.prime }) * n * z
        >>> w.space
        |b,b',c>
        >>> w.trace({ hb: hb.prime }).space
        |c>
        >>> w.trace({ hb: hb.prime }).closeto(z)
        True
        >>> m = ha.random_array().normalized()
        >>> v = w * m * m.H.relabel(ha.H, hc.H)
        >>> v.space
        |a,b,b',c><c|
        >>> v.trace({ hb: hb.prime, hc.H: ha }).space
        |c>
        >>> v.trace({ hb: hb.prime, hc.H: ha }).closeto(z)
        True
        """

        if axes is None:
            if self.space != self.space.H:
                raise HilbertError('bra space does not equal ket space; '+
                    'please specify axes')
            # The full trace is handled specially here, for efficiency.
            return np.trace( self.as_np_matrix() )

        if isinstance(axes, HilbertSpace):
            axes = axes.bra_ket_set
            # and then process further in the next if block

        if not isinstance(axes, dict):
            axes_set = set()
            for s in HilbertSpace._expand_list_to_atoms(list(axes)):
                if not s in self.space.bra_ket_set:
                    raise HilbertIndexError('not in ket set: '+repr(s))
                axes_set.add(s.H if s.is_dual else s)
            axes = dict((s, s.H) for s in axes_set)

        assert isinstance(axes, dict)

        HilbertSpace._assert_nodup_space(axes.keys()+axes.values(), "a space was listed twice")

        for (k, v) in axes.iteritems():
            if not k in self.space.bra_ket_set:
                raise HilbertIndexError("not in this array's space: "+repr(k))
            if not v in self.space.bra_ket_set:
                raise HilbertIndexError("not in this array's space: "+repr(v))

        # The full trace is handled specially here, for efficiency.
        if frozenset(axes.keys()+axes.values()) == self.space.bra_ket_set:
            return np.trace( self.as_np_matrix() )

        working = self

        for (s1, s2) in axes.iteritems():
            axis1 = working.get_dim(s1)
            axis2 = working.get_dim(s2)

            arr = np.trace( working.nparray, axis1=axis1, axis2=axis2 )

            out_space = working.space.bra_ket_set - frozenset((s1, s2))

            if len(out_space) == 0:
                # arr should be a scalar
                working = arr
            else:
                out_space = self.space.base_field.create_space1(out_space)
                working = out_space.array(arr)

        return working

    def expm(self, q=7):
        """
        Return the matrix exponential of this array.

        It is required that the dimension of the bra space be equal to the
        dimension of the ket space.

        :param q: order of the Pade approximation (see the scipy.linalg.expm
            documentation for details)
        :type q: integer; default 7

        >>> import numpy
        >>> from qitensor import qubit
        >>> numpy.set_printoptions(suppress = True)
        >>> ha = qubit('a')
        >>> (ha.X * numpy.pi * 1j).expm()
        HilbertArray(|a><a|,
        array([[-1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]]))
        """

        return self.np_matrix_transform( \
            lambda x: self.space.base_field.mat_expm(x, q))

    def svd(self, full_matrices=True, inner_space=None):
        """
        Return the singular value decomposition of this array.

        :param full_matrices: if True, U and V are square.  If False,
            S is square.
        :type full_matrices: bool; default True
        :param inner_space: Hilbert space for S.
        :type inner_space: HilbertSpace

        x.svd() returns a tuple (U, S, V) such that:
        * ``x == U * S * V``
        * ``U.H * U`` is identity
        * ``S`` is diagonal
        * ``V * V.H`` is identity

        If full_matrices is True:

        * U and V will be square.

        * If inner_space is None, the bra and ket spaces of U will be the same
          as the ket space of the input and the bra and ket spaces of V will be
          the same as the bra space of the input.  The Hilbert space of S will be
          the same as that of the input.  If the input is not square (the
          dimension of the bra space does not match that of the ket space) then S
          will not be square.

        * If inner_space is not None, it should be a HilbertSpace whose bra and
          ket dimensions are the same as those of the input.

        If full_matrices is False:

        * S will be square.  One of U or V will be square.

        * If inner_space is None, the bra and ket spaces of S will be the same.
          Either the bra or the ket space of the input will be used for S,
          whichever is of smaller dimension.  If they are of equal dimension
          but are not the same spaces, then there is an ambiguity and an
          exception will be raised.  In this case, you must manually specify
          inner_space.

        * If inner_space is not None, it should be a ket space, and must be
          of the same dimension as the smaller of the bra or ket spaces of the
          input.  The given space will be used for both the bra and the ket
          space of S.

        See also: :func:`singular_vals`, :func:`svd_list`

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> x = (ha * hb.H * hc.H).random_array()
        >>> x.space
        |a><b,c|

        >>> (U, S, V) = x.svd()
        >>> [h.space for h in (U, S, V)]
        [|a><a|, |a><b,c|, |b,c><b,c|]
        >>> (U * S * V - x).norm() < 1e-14
        True

        >>> (U, S, V) = x.svd(full_matrices=False)
        >>> [h.space for h in (U, S, V)]
        [|a><a|, |a><a|, |a><b,c|]
        >>> (U * S * V - x).norm() < 1e-14
        True

        >>> hS = qubit('d1') * qudit('d2', 4).H
        >>> hS
        |d1><d2|
        >>> (U, S, V) = x.svd(full_matrices=True, inner_space=hS)
        >>> [h.space for h in (U, S, V)]
        [|a><d1|, |d1><d2|, |d2><b,c|]
        >>> (U * S * V - x).norm() < 1e-14
        True

        >>> hS = qubit('d')
        >>> (U, S, V) = x.svd(full_matrices=False, inner_space=hS)
        >>> [h.space for h in (U, S, V)]
        [|a><d|, |d><d|, |d><b,c|]
        >>> (U * S * V - x).norm() < 1e-14
        True
        """

        hs = self.space

        if inner_space is None:
            if full_matrices:
                inner_space = hs
            else:
                bs = hs.bra_space()
                ks = hs.ket_space()
                bra_size = shape_product(bs.shape)
                ket_size = shape_product(ks.shape)
                if ks == bs:
                    inner_space = ks
                elif bra_size < ket_size:
                    inner_space = bs.H
                elif ket_size < bra_size:
                    inner_space = ks
                else:
                    # Ambiguity as to which space to take, force user to
                    # specify.
                    raise HilbertError('Please specify which Hilbert space to '+
                        'use for the singular values of this square matrix')

        if not isinstance(inner_space, HilbertSpace):
            raise TypeError('inner_space must be a HilbertSpace')

        (u, s, v) = hs.base_field.mat_svd(self.as_np_matrix(), full_matrices)

        if full_matrices:
            u_space = hs.ket_space() * inner_space.ket_space().H
            v_space = hs.bra_space() * inner_space.bra_space().H
            U = u_space.reshaped_np_matrix(u)
            V = v_space.reshaped_np_matrix(v)

            dim1 = shape_product(inner_space.ket_space().shape)
            dim2 = shape_product(inner_space.bra_space().shape)
            min_dim = np.min([dim1, dim2])
            Sm = np.zeros((dim1, dim2), dtype=hs.base_field.dtype)
            Sm[:min_dim, :min_dim] = np.diag(s)
            S = inner_space.reshaped_np_matrix(Sm)
        else:
            inner_space.assert_ket_space()

            u_space = hs.ket_space() * inner_space.H
            v_space = inner_space * hs.bra_space()
            U = u_space.reshaped_np_matrix(u)
            V = v_space.reshaped_np_matrix(v)

            s_mat_space = inner_space * inner_space.H
            S = s_mat_space.diag(s)

        return (U, S, V)

    def svd_list(self, row_space=None, col_space=None, thresh=0):
        # FIXME - docs
        """
        >>> from qitensor import qubit, qudit
        >>> ha = qudit('a', 3)
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> W = (ha * hb.H * hc.H).random_array()
        >>> W.space
        |a><b,c|

        >>> # test basic properties of SVD
        >>> import numpy as np
        >>> (Ul, sl, Vl) = W.svd_list()
        >>> (len(Ul), len(sl), len(Vl))
        (3, 3, 3)
        >>> (Ul[0].space, Vl[0].space)
        (|a>, <b,c|)
        >>> np.allclose(np.array([[x.H*y for x in Ul] for y in Ul]), np.eye(len(sl)))
        True
        >>> np.allclose(np.array([[x*y.H for x in Vl] for y in Vl]), np.eye(len(sl)))
        True
        >>> (np.sum([u*s*v for (u,s,v) in zip(Ul, sl, Vl)]) - W).norm() < 1e-14
        True

        >>> # take SVD across the |a><b| vs. <c| cut
        >>> import numpy
        >>> (Ul, sl, Vl) = W.svd_list(col_space=ha*hb.H)
        >>> (len(Ul), len(sl), len(Vl))
        (2, 2, 2)
        >>> (Ul[0].space, Vl[0].space)
        (|a><b|, <c|)
        >>> np.allclose(np.array([[(x.H*y).trace() for x in Ul] for y in Ul]), np.eye(len(sl)))
        True
        >>> np.allclose(np.array([[x*y.H for x in Vl] for y in Vl]), np.eye(len(sl)))
        True
        >>> (np.sum([u*s*v for (u,s,v) in zip(Ul, sl, Vl)]) - W).norm() < 1e-14
        True

        >>> # as above, but with col_space given as a list
        >>> (Ul, sl, Vl) = W.svd_list(col_space=[hb.H, ha])
        >>> (Ul[0].space, Vl[0].space)
        (|a><b|, <c|)
        >>> (np.sum([u*s*v for (u,s,v) in zip(Ul, sl, Vl)]) - W).norm() < 1e-14
        True
        """

        hs = self.space

        rowcol_kw = { 'row_space': row_space, 'col_space': col_space }
        (row_space, col_space) = self._get_row_col_spaces(**rowcol_kw)
        assert len(row_space) > 0
        assert len(col_space) > 0
        m = self.as_np_matrix(**rowcol_kw)

        (u, s, v) = hs.base_field.mat_svd(m, False)
        #print u.shape
        #print s.shape
        #print v.shape
        #print row_space
        #print col_space

        U_list = np.array([ \
            np.product(col_space).array(x, reshape=True, input_axes=col_space) \
            for x in u.T])
        V_list = np.array([ \
            np.product(row_space).array(x, reshape=True, input_axes=row_space) \
            for x in v])

        if thresh > 0:
            U_list = U_list[s > thresh]
            V_list = V_list[s > thresh]
            s = s[s > thresh]

        return (U_list, s, V_list)

    def singular_vals(self, row_space=None, col_space=None):
        """
        Returns the singular values of this array.

        See also: :func:`svd`, :func:`svd_list`

        >>> from qitensor import qubit, qudit
        >>> import numpy
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> x = (ha * hb.H * hc.H).random_array()
        >>> numpy.allclose(numpy.diag(x.svd()[1].as_np_matrix()), x.singular_vals())
        True
        """

        rowcol_kw = { 'row_space': row_space, 'col_space': col_space }
        m = self.as_np_matrix(**rowcol_kw)

        return self.space.base_field.mat_svd_vals(m)

    def eig(self, w_space=None, hermit=False):
        """
        Return the eigenvalues and right eigenvectors of this array.

        :param w_space: space for the diagonal matrix, if None the space of the
            input array is used.
        :type w_space: HilbertSpace; default None
        :param hermit: set this to True if the input is Hermitian
        :type hermit: bool; default False

        NOTE: in the case of degenerate eigenvalues, with hermit=False, it may
        be the case that the returned eigenvectors array is not full rank.  See
        the documentation for numpy.linalg.eig for details.

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> hc = qudit('c', 6)
        >>> epsilon = 1e-13

        >>> op = (ha*hb).O.random_array()
        >>> # make a normal operator
        >>> op = op.H * op
        >>> (W, V) = op.eig()
        >>> V.space
        |a,b><a,b|
        >>> W.space
        |a,b><a,b|
        >>> (V.H * V - (ha*hb).eye()).norm() < epsilon
        True
        >>> (V.H * op * V - W).norm() < epsilon
        True
        >>> (op * V - V * W).norm() < epsilon
        True

        >>> # NOTE: this is not a normal operator, so V won't be unitary.
        >>> op = (ha*hb).O.random_array()
        >>> (W, V) = op.eig(w_space=hc)
        >>> V.space
        |a,b><c|
        >>> W.space
        |c><c|
        >>> (op * V - V * W).norm() < epsilon
        True

        >>> vec = hb.random_array().normalized()
        >>> dyad = vec * vec.H
        >>> (W, V) = dyad.eig(hermit=True)
        >>> (W - hb.diag([1, 0, 0])).norm() < epsilon
        True
        >>> (V.H * V - hb.eye()).norm() < epsilon
        True
        >>> vec2 = V[:, 0]
        >>> # Correct for phase ambiguity
        >>> vec2 *= (vec[0]/vec2[0]) / abs(vec[0]/vec2[0])
        >>> (vec - vec2).norm() < epsilon
        True
        """

        if not self.space.is_symmetric():
            raise HilbertError('bra space must be the same as ket space '+
                '(space was '+repr(self.space)+')')

        if w_space is None:
            w_space = self.space.ket_space()

        w_space.assert_ket_space()

        (w, v) = self.space.base_field.mat_eig(self.as_np_matrix(), hermit)

        # sort eigenvalues in ascending order of real component
        srt = np.argsort(-w)
        w = w[srt]
        v = v[:, srt]

        W = (w_space * w_space.H).diag(w)
        V = (self.space.ket_space() * w_space.H).reshaped_np_matrix(v)
        return (W, V)

    def eigvals(self, hermit=False):
        """
        Return the eigenvalues of this array, sorted in order of decreasing
        real component.

        :param hermit: set this to True if the input is Hermitian.  In this
            case, the returned eigenvalues will be real.
        :type hermit: bool; default False

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> epsilon = 1e-13

        >>> op = (ha*hb).O.random_array()
        >>> # make a normal operator
        >>> op = op.H * op
        >>> (W1, V1) = op.eig()
        >>> W2 = op.eigvals()
        >>> (ha*hb).diag(W2) == W1
        True
        """

        if not self.space.is_symmetric():
            raise HilbertError('bra space must be the same as ket space '+
                '(space was '+repr(self.space)+')')

        w = self.space.base_field.mat_eigvals(self.as_np_matrix(), hermit)

        # sort eigenvalues in ascending order of real component
        w = -np.sort(-w)

        if hermit:
            assert np.all(np.imag(w) == 0)
            w = np.real(w)

        return w

    def sqrt(self):
        """
        Return the square root of this matrix.

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> P = (ha*hb).O.random_array()
        >>> # make a positive operator
        >>> P = P.H * P
        >>> P.space
        |a,b><a,b|
        >>> Q = P.sqrt()
        >>> Q.space
        |a,b><a,b|
        >>> P.closeto(Q.H * Q)
        True
        >>> P.closeto(Q * Q.H)
        True
        """

        if not self.space.is_symmetric():
            raise HilbertError('bra space must be the same as ket space '+
                '(space was '+repr(self.space)+')')

        if not self.closeto(self.H):
            raise HilbertError('matrix was not Hermitian')

        (W, V) = self.eig(hermit=True)
        W = np.diag(W.as_np_matrix())
        if not np.all(W >= 0):
            raise HilbertError('matrix was not positive')
        W = self.space.diag(np.sqrt(W))
        return V * W * V.H

    def entropy(self, normalize=False, checks=True):
        """
        Returns the von Neumann entropy of a density operator, in bits.

        :param normalize: if True, the input is automatically normalized to
            trace one.  If false, an exception is raised if the trace is not
            one.
        :type normalize: bool; default False
        :param checks: if False, don't check that the input is a valid density
            matrix or Hermitian.  This is sometimes needed for symbolic
            computations.
        :type checks: bool; default True

        >>> import numpy as np
        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> # entropy of a pure state is zero
        >>> ha.ket(0).O.entropy()
        0.0
        >>> # a fully mixed state of dimension 2
        >>> (ha.ket(0).O/2 + ha.ket(1).O/2).entropy()
        1.0
        >>> # a fully mixed state of dimension 3
        >>> abs( (hb.eye()/3).entropy() - np.log2(3) ) < 1e-10
        True
        >>> # automatic normalization
        >>> abs( hb.eye().entropy(normalize=True) - np.log2(3) ) < 1e-10
        True
        >>> # a bipartite pure state
        >>> s = (ha.ket(0) * hb.array([1/np.sqrt(2),0,0]) + ha.ket(1) * hb.array([0,0.5,0.5]))
        >>> np.round(s.O.entropy(), 10)
        0.0
        >>> # entanglement across the a-b cut
        >>> (s.O.trace(hb)).entropy()
        1.0
        >>> # entanglement across the a-b cut is the same as across b-a cut
        >>> s = (ha*hb).random_array().normalized().O
        >>> abs(s.trace(ha).entropy() - s.trace(hb).entropy()) < 1e-10
        True
        """

        if not self.space.is_symmetric():
            raise HilbertError("bra and ket spaces must be the same")

        if checks and not (self == self.H or np.allclose(self.nparray, self.H.nparray)):
            raise HilbertError("density matrix must be Hermitian")

        norm = self.trace()
        if normalize:
            densmat = self / norm
        else:
            if checks and abs(norm-1) > 1e-9:
                raise HilbertError('density matrix was not normalized: norm='+str(norm))
            densmat = self

        schmidt = densmat.eigvals(hermit=True)

        if checks:
            # should have been taken care of by normalization above
            assert abs(sum(schmidt)-1) < 1e-9

            if not np.all(schmidt >= -1e-9):
                raise HilbertError('density matrix was not positive: '+str(schmidt))

        return sum([-self.space.base_field.xlog2x(x) for x in schmidt])

    def QR(self, inner_space=None):
        """
        Returns operators Q and R such that Q is an isometry, R is upper triangular, and self=Q*R.

        The bra space of Q (and the ket space of R) is the smaller of the bra or ket spaces of
        the input.  This can be overridden using the inner_space parameter.

        :param inner_space: bra space of Q (and the ket space of R)
        :type inner_space: HilbertSpace

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> hc = qubit('c')
        >>> m = (hb*hc*ha.H).random_array()
        >>> (q, r) = m.QR()
        >>> (q.space, r.space)
        (|b,c><a|, |a><a|)
        >>> (q.H * q - ha.eye()).norm() < 1e-14
        True
        >>> (q*r - m).norm() < 1e-14
        True
        >>> (q, r) = ha.O.random_array().QR(inner_space=hb)
        >>> (q.space, r.space)
        (|a><b|, |b><a|)
        """

        hs = self.space
        mat = self.as_np_matrix()

        (q, r) = self.space.base_field.mat_qr(mat)

        if inner_space is None:
            if mat.shape[0] < mat.shape[1]:
                inner_space = hs.ket_space()
            else:
                inner_space = hs.bra_space().H

        inner_space.assert_ket_space()

        Q = (hs.ket_space() * inner_space.H).reshaped_np_matrix(q)
        R = (inner_space * hs.bra_space()).reshaped_np_matrix(r)

        return (Q, R)

    def span(self, axes='all'):
        """
        Returns a TensorSubspace for the column/row/mixed space of this array (EXPERIMENTAL).

        :param axes: the axes to take the span of
        :type new_data: string ('all', 'col', 'row') or HilbertSpace

        >>> from qitensor import qudit
        >>> ha = qudit('a', 2)
        >>> hb = qudit('b', 3)
        >>> hc = qudit('c', 3)
        >>> iso = (hb*ha.H).random_isometry()
        >>> proj = iso * iso.H
        >>> proj.span('col')
        <TensorSubspace of dim 2 over space (|b>)>
        >>> bigop = proj * ha.random_array() * hc.H.random_array()
        >>> bigop.space
        |a,b><b,c|
        >>> bigop.span('col')
        <TensorSubspace of dim 2 over space (|a,b>)>
        >>> bigop.span(hb.O)
        <TensorSubspace of dim 1 over space (|b><b|)>
        """
        # FIXME - not implemented
        # >>> bigop.span(hb.O).tensor_prod(bigop.span(ha))
        # <TensorSubspace of dim 2 over space (|a,b><b|)>

        if axes == 'all':
            axes = self.space
        elif axes == 'col':
            axes = self.space.ket_space()
        elif axes == 'row':
            axes = self.space.bra_space()

        assert isinstance(axes, HilbertSpace)
        assert axes.bra_ket_set <= self.space.bra_ket_set

        space_axes = [i for (i,s) in enumerate(self.axes) if s in axes.bra_ket_set]
        group_axes = [i for (i,s) in enumerate(self.axes) if s not in axes.bra_ket_set]
        group_dim = self.space.dim() // axes.dim()
        assert len(group_axes) + len(space_axes) == len(self.axes)

        v = self.nparray.transpose(group_axes+space_axes)
        v = v.reshape((group_dim,) + axes.shape)
        import qitensor.experimental.subspace
        return qitensor.experimental.subspace.TensorSubspace.from_span(v, hilb_space=axes)

    ########## stuff that only works in Sage ##########

    def n(self, prec=None, digits=None):
        """
        Converts symbolic values to numeric values (only useful in Sage).
        """

        return self.np_matrix_transform( \
            lambda x: self.space.base_field.mat_n(x, prec, digits))

    def simplify(self):
        """
        Simplifies symbolic expressions (only useful in Sage).
        """

        return self.space.base_field.mat_simplify(self)

    def simplify_full(self):
        """
        Simplifies symbolic expressions (only useful in Sage).
        """

        return self.space.base_field.mat_simplify(self, full=True)

    def _matrix_(self, R=None):
        return self.sage_matrix(R)

    def sage_matrix(self, R=None):
        if not have_sage:
            raise HilbertError('This is only available under Sage')

        return self.space.base_field.matrix_np_to_sage( \
            self.as_np_matrix(), R)

    def _latex_(self):
        return FORMATTER.array_latex_block_table(self, use_hline=False)

    def sage_block_matrix(self, R=None):
        if not have_sage:
            raise HilbertError('This is only available under Sage')

        hs = self.space

        blocks = [self]
        nrows = 1
        ncols = 1

        if len(hs.sorted_kets) > 1:
            h = hs.sorted_kets[0]
            blocks = [m[{h: i}] for m in blocks for i in h.indices]
            nrows = len(h.indices)

        if len(hs.sorted_bras) > 1:
            h = hs.sorted_bras[0]
            blocks = [m[{h: i}] for m in blocks for i in h.indices]
            ncols = len(h.indices)

        blocks = [x.sage_matrix(R=R) for x in blocks]

        import sage.all

        return sage.all.block_matrix(blocks, nrows=nrows, ncols=ncols, subdivide=True)

    def sage_matrix_transform(self, f, transpose_dims=False):
        if not have_sage:
            raise HilbertError('This is only available under Sage')

        out_hilb = self.space
        if transpose_dims:
            out_hilb = out_hilb.H

        m = self.sage_matrix()
        m = f(m)
        return out_hilb.reshaped_sage_matrix(m)

    def __str__(self):
        return FORMATTER.array_str(self)

    def __repr__(self):
        return FORMATTER.array_repr(self)

    ########## IPython stuff ##########

    def _repr_latex_(self):
        if not FORMATTER.ipy_table_format_mode == 'latex':
            return None
        latex = FORMATTER.array_latex_block_table(self, use_hline=True)
        return '$$'+latex+'$$'

    def _repr_html_(self):
        if not FORMATTER.ipy_table_format_mode == 'html':
            return None
        return FORMATTER.array_html_block_table(self)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
