import numpy as np

from qitensor.exceptions import *
from qitensor.space import *

class HilbertArray(object):
    # only called from HilbertSpace.array
    def __init__(self, space, data, noinit_data, reshape):
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
        ret = self.space.array(noinit_data=True)
        ret.nparray = self.nparray.copy()
        return ret

    def _reassign(self, other):
        self.space = other.space
        self.nparray = other.nparray
        self.axes = other.axes

    def get_dim(self, simple_hilb):
        return self.axes.index(simple_hilb)

    def assert_same_axes(self, other):
        if self.axes != other.axes:
            raise HilbertIndexError('Mismatched HilbertSpaces: '+
                repr(self.space)+' vs. '+repr(other.space))

    def set_data(self, new_data):
        if isinstance(new_data, HilbertArray):
            self.assert_same_axes(new_data)
            self.set_data(new_data.nparray)
        else:
            # This is needed to make slices work properly
            self.nparray[:] = new_data

    def tensordot(self, other, contraction_spaces=None):
        hs = self.space
        ohs = other.space
        #print str(hs)+'*'+str(ohs)

        hs.base_field.assert_same(ohs.base_field)

        if contraction_spaces is None:
            mul_space = frozenset([x.H for x in hs.bra_set]) & ohs.ket_set
        elif isinstance(contraction_spaces, frozenset):
            mul_space = contraction_spaces
        elif isinstance(contraction_spaces, HilbertSpace):
            mul_space = contraction_spaces.ket_set
            contraction_spaces.assert_ket_space()
        else:
            raise TypeError('contraction space must be HilbertSpace '+
                'or frozenset')

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

        ret = ret_space.array(noinit_data=True)
        #print "ret", ret.axes
        permute = tuple([td_axes.index(x) for x in ret.axes])
        #print permute
        ret.nparray = td.transpose(permute)

        return ret

    def transpose(self, tpose_axes=None):
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
        self.assert_same_axes(other)
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
        self.assert_same_axes(other)
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
        ket_size = np.product([len(x.indices) \
            for x in self.axes if not x.is_dual])
        bra_size = np.product([len(x.indices) \
            for x in self.axes if x.is_dual])
        assert ket_size * bra_size == np.product(self.nparray.shape)
        return np.matrix(self.nparray.reshape(ket_size, bra_size))

    def np_matrix_transform(self, f, transpose_dims=False):
        m = self.as_np_matrix()
        m = f(m)
        out_hilb = self.space
        if transpose_dims:
            out_hilb = out_hilb.H
        return out_hilb.reshaped_np_matrix(m)

    @property
    def H(self):
        return self.space.base_field.mat_adjoint(self)

    @property
    def I(self):
        return self.space.base_field.mat_inverse(self)

    @property
    def T(self):
        # transpose should be the same for all base_field's
        return self.np_matrix_transform(lambda x: x.T, transpose_dims=True)

    def det(self):
        return self.space.base_field.mat_det(self)

    def fill(self, val):
        # fill should be the same for all base_field's
        self.nparray.fill(val)

    def norm(self, ord=None):
        return self.space.base_field.mat_norm(self, ord)

    def normalize(self):
        """Normalizes array in-place."""
        self /= self.norm()
        return self

    def normalized(self):
        """Returns a normalized copy of array."""
        return self / self.norm()

    def pinv(self, rcond=1e-15):
        return self.space.base_field.mat_pinv(self, rcond)

    def conj(self):
        return self.space.base_field.mat_conj(self)

    def expm(self, q=7):
        return self.space.base_field.mat_expm(self, q)

    def svd(self, inner_spaces=None, full_matrices=True):
        if inner_spaces is None:
            hs = self.space
            bs = hs.bra_space()
            ks = hs.ket_space()
            if full_matrices:
                inner_spaces = (ks, bs.H)
            else:
                bra_size = np.product(bs.shape)
                ket_size = np.product(ks.shape)
                if ks == bs:
                    inner_spaces = (ks,)
                elif bra_size < ket_size:
                    inner_spaces = (bs.H,)
                elif ket_size < bra_size:
                    inner_spaces = (ks,)
                else:
                    # Ambiguity as to which space to take, force user to
                    # specify.
                    raise HilbertError('Please specify which Hilbert space to '+
                        'use for the singular values of this square matrix')

        if not isinstance(inner_spaces, tuple):
            raise TypeError('inner_spaces must be a tuple')

        if full_matrices:
            if len(inner_spaces) != 2:
                raise ValueError('full_matrices=True requires inner_spaces to '+
                    'be a tuple of length 2')
            (u_inner_space, v_inner_space) = inner_spaces
            u_inner_space.assert_ket_space()
            v_inner_space.assert_ket_space()
            return self.space.base_field.mat_svd_full(
                self, u_inner_space, v_inner_space)
        else:
            if len(inner_spaces) != 1:
                raise ValueError('full_matrices=True requires inner_spaces to '+
                    'be a tuple of length 1')
            (s_space,) = inner_spaces
            return self.space.base_field.mat_svd_partial(self, s_space)
