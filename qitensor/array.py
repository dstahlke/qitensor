"""
A HilbertArray is a vector in a HilbertSpace.  Internally, it is backed by a
numpy.array.  HilbertArray's are to be created using the
:meth:`HilbertSpace.array` method.
"""

import numpy as np

from qitensor.exceptions import *
from qitensor.space import HilbertSpace
from qitensor.atom import HilbertAtom

__all__ = ['HilbertArray']

class HilbertArray(object):
    def __init__(self, space, data, noinit_data, reshape):
        """
        Don't call this constructor yourself, use HilbertSpace.array
        """

        hs = space
        self.space = hs

        if noinit_data:
            self.nparray = None
        elif data is None:
            self.nparray = np.zeros(hs.shape, dtype=hs.base_field.dtype)
        else:
            self.nparray = np.array(data, dtype=hs.base_field.dtype)
            if reshape:
                if np.product(self.nparray.shape) != np.product(hs.shape):
                    raise HilbertShapeError(np.product(data.shape), 
                        np.product(hs.shape))
                self.nparray = self.nparray.reshape(hs.shape)
            if self.nparray.shape != hs.shape:
                raise HilbertShapeError(self.nparray.shape, hs.shape)

        self.axes = hs.sorted_kets + hs.sorted_bras

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
        self.space = other.space
        self.nparray = other.nparray
        self.axes = other.axes

    def get_dim(self, simple_hilb):
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

        return self.axes.index(simple_hilb)

    def _assert_same_axes(self, other):
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
        """

        hs = self.space
        ohs = other.space
        #print str(hs)+'*'+str(ohs)

        hs.base_field.assert_same(ohs.base_field)

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

        ret_space = \
            hs.base_field.create_space2(hs.ket_set, (hs.bra_set-mul_H)) * \
            hs.base_field.create_space2((ohs.ket_set-mul_space), ohs.bra_set)
        #print 'ret', ret_space

        axes_self  = [self.get_dim(x.H) for x in sorted(mul_space)]
        axes_other = [other.get_dim(x)  for x in sorted(mul_space)]
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
        td_axes = []
        td_axes += [x for x in hs.sorted_kets]
        td_axes += [x for x in hs.sorted_bras if not x in mul_H]
        td_axes += [x for x in ohs.sorted_kets if not x in mul_space]
        td_axes += [x for x in ohs.sorted_bras]
        #print td_axes
        #print td.shape
        assert len(td_axes) == len(td.shape)

        if len(td_axes) == 0:
            # convert 0-d array to scalar
            return td[()]
        else:
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

    def relabel(self, from_space, to_space):
        """
        Changes the HilbertSpace of this array without changing data.

        :param from_space: the old space
        :type from_space: HilbertSpace
        :param to_space: the new space
        :type to_space: HilbertSpace

        Either ``from_space`` and ``to_space`` should both be bra spaces
        or both should be ket spaces.

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
        """

        # FIXME - this could be made a lot more efficient by not doing a
        # multiplication operation.
        if len(from_space.ket_set) == 0 and len(to_space.ket_set) == 0:
            # we were given bra spaces
            return self * (from_space.H * to_space).eye()
        elif len(from_space.bra_set) == 0 and len(to_space.bra_set) == 0:
            # we were given ket spaces
            return (to_space * from_space.H).eye() * self
        else:
            # Maybe the mixed bra/ket case could do a partial transpose.
            raise BraKetMixtureError('from_space and to_space must both be '+
                'bras or both be kets')

    def __eq__(self, other):
        if self.space != other.space:
            return False
        else:
            return np.all(self.nparray == other.nparray)

    def __ne__(self, other):
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
        if isinstance(other, HilbertArray):
            return self.tensordot(other)
        else:
            ret = self.copy()
            ret *= other
            return ret

    def __imul__(self, other):
        if isinstance(other, HilbertArray):
            self._reassign(self * other)
        else:
            # hopefully other is a scalar
            self.nparray *= other
        return self

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if not isinstance(other, HilbertArray):
            raise TypeError('HilbertArray can only add another HilbertArray')
        ret = self.copy()
        ret += other
        return ret

    def __iadd__(self, other):
        if not isinstance(other, HilbertArray):
            raise TypeError('HilbertArray can only add another HilbertArray')
        self._assert_same_axes(other)
        self.nparray += other.nparray
        return self

    def __sub__(self, other):
        if not isinstance(other, HilbertArray):
            raise TypeError('HilbertArray can only subtract '+
                'another HilbertArray')
        ret = self.copy()
        ret -= other
        return ret

    def __isub__(self, other):
        if not isinstance(other, HilbertArray):
            raise TypeError('HilbertArray can only subtract '+
                'another HilbertArray')
        self._assert_same_axes(other)
        self.nparray -= other.nparray
        return self

    def __div__(self, other):
        ret = self.copy()
        ret /= other
        return ret

    def __idiv__(self, other):
        self.nparray /= other
        return self

    def __truediv__(self, other):
        ret = self.copy()
        ret.__itruediv__(other)
        return ret

    def __itruediv__(self, other):
        self.nparray.__itruediv__(other)
        return self

    def __str__(self):
        return str(self.space) + '\n' + str(self.nparray)

    def __repr__(self):
        return 'HilbertArray('+repr(self.space)+',\n'+ \
            repr(self.nparray)+')'

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
                if not isinstance(k, slice):
                    index_map[self.axes[i]] = k
        else:
            if len(self.axes) != 1:
                raise HilbertIndexError("Wrong number of indices given "+
                    "(1 for %s)" % str(self.space))
            if not isinstance(key, slice):
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

    def as_np_matrix(self):
        """
        Returns the underlying data as a numpy.matrix.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> x = (ha.O * hb).random_array()
        >>> x.space
        |a,b><a|
        >>> x.as_np_matrix().shape
        (4, 2)
        """

        ket_size = np.product([len(x.indices) \
            for x in self.axes if not x.is_dual])
        bra_size = np.product([len(x.indices) \
            for x in self.axes if x.is_dual])
        assert ket_size * bra_size == np.product(self.nparray.shape)
        return np.matrix(self.nparray.reshape(ket_size, bra_size))

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

        return self.space.base_field.mat_adjoint(self)

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

        return self.space.base_field.mat_inverse(self)

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

        return self.space.base_field.mat_det(self)

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

        return self.space.base_field.mat_norm(self)

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

        return self.space.base_field.mat_pinv(self, rcond)

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

        return self.space.base_field.mat_conj(self)

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
        >>> ha = qubit('a')
        >>> (ha.X * numpy.pi * 1j).expm()
        HilbertArray(|a><a|,
        array([[-1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]]))
        """

        return self.space.base_field.mat_expm(self, q)

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

        if inner_space is None:
            hs = self.space
            if full_matrices:
                inner_space = hs
            else:
                bs = hs.bra_space()
                ks = hs.ket_space()
                bra_size = np.product(bs.shape)
                ket_size = np.product(ks.shape)
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
            raise TypeError('inner_space must be HilbertSpace')

        if full_matrices:
            return self.space.base_field.mat_svd_full(
                self, inner_space)
        else:
            inner_space.assert_ket_space()
            return self.space.base_field.mat_svd_partial(self, inner_space)
