# FIXME - cannot set slice data e.g. m[0:3] = [1,2,3]
# FIXME - use singletons for HilbertAtom and HilbertBaseField (and make pickle restore the singletons)

import numpy as np
import numpy.linalg as linalg
import numpy.random

class HilbertError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

class MismatchedIndexSetError(HilbertError):
    def __init__(self, msg):
        HilbertError.__init__(self, msg)

class DuplicatedSpaceError(HilbertError):
    def __init__(self, spaces):
        HilbertError.__init__(self, repr(spaces))

class BraKetMixtureError(HilbertError):
    def __init__(self, msg):
        HilbertError.__init__(self, msg)

class HilbertIndexError(HilbertError, LookupError):
    def __init__(self, msg):
        HilbertError.__init__(self, msg)
        LookupError.__init__(self, )

class IncompatibleBaseFieldError(HilbertError):
    def __init__(self, msg):
        HilbertError.__init__(self, msg)

class HilbertShapeError(HilbertError, ValueError):
    def __init__(self, shape1, shape2):
        msg = repr(shape1)+' vs. '+repr(shape2)
        HilbertError.__init__(self, msg)
        ValueError.__init__(self, )

class NotKetSpaceError(HilbertError):
    def __init__(self, msg):
        HilbertError.__init__(self, msg)

######################################################################

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

    def mat_norm(self, m, ord=None):
        return linalg.norm(m.nparray)

    def mat_pinv(self, m, rcond):
        return m.np_matrix_transform(
            lambda x: np.linalg.pinv(x, rcond), transpose_dims=True)

    def mat_conj(self, m):
        return m.np_matrix_transform(lambda x: x.conj())

    def mat_expm(self, m, q):
        import scipy.linalg
        return m.np_matrix_transform(lambda x: scipy.linalg.expm(x, q))

    def mat_svd_full(self, m, u_inner_space, v_inner_space):
        (u, s, v) = linalg.svd(m.as_np_matrix())
        u_space = m.space.ket_space() * u_inner_space.H
        s_space = u_inner_space * v_inner_space.H
        v_space = v_inner_space * m.space.bra_space()
        U = u_space.reshaped_np_matrix(u)
        V = v_space.reshaped_np_matrix(v)

        dim1 = np.product(u_inner_space.shape)
        dim2 = np.product(v_inner_space.shape)
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

        >>> from qitensor import *
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> ha * hb == complex_field.create_space1([ha, hb])
        True
        >>> ha.H * hb == complex_field.create_space1([ha.H, hb])
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

        >>> from qitensor import *
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> ha * hb == complex_field.create_space2(frozenset([ha, hb]), frozenset())
        True
        >>> ha.H * hb == complex_field.create_space2(frozenset([hb]), frozenset([ha.H]))
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

        See also: :func:`qubit`, :func:`qudit`

        >>> from qitensor import *
        >>> ha = indexed_space('a', ['x', 'y', 'z'])
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

        See also: :func:`qubit`, :func:`indexed_space`

        >>> from qitensor import *
        >>> ha = qudit('a', 3)
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

        See also: :func:`qudit`, :func:`indexed_space`

        >>> from qitensor import *
        >>> ha = qubit('a')
        >>> ha
        |a>
        >>> ha.indices
        [0, 1]
        """

        return self.qudit(label, 2, latex_label)

class HilbertSpace(object):
    def __init__(self, ket_set, bra_set, base_field):
        """
        Constructor should only be called from :func:`HilbertBaseField._space_factory`
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

        >>> from qitensor import *
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> (ha.H * hb).bra_space()
        <a|
        """
        return self.base_field.create_space2(frozenset(), self.bra_set)

    def ket_space(self):
        """
        Returns a ``HilbertSpace`` consisting of only the ket space of this
        space.

        >>> from qitensor import *
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> (ha.H * hb).ket_space()
        |b>
        """
        return self.base_field.create_space2(self.ket_set, frozenset())

    @property
    def H(self):
        if self._H is None:
            self._H = self.base_field.create_space1(
                [x.H for x in self.bra_ket_set])
        return self._H

    @property
    def O(self):
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

    def reshaped_np_matrix(self, m):
        ket_size = np.product([len(x.indices) for x in self.ket_set])
        bra_size = np.product([len(x.indices) for x in self.bra_set])
        if m.shape[0] != ket_size:
            raise HilbertShapeError(m.shape[0], ket_size)
        if m.shape[1] != bra_size:
            raise HilbertShapeError(m.shape[1], bra_size)
        return self.array(m, reshape=True)

    def array(self, data=None, noinit_data=False, reshape=False):
        return self.base_field._array_factory(self, data, noinit_data, reshape)

    def random_array(self):
        return self.array(self.base_field.random_array(self.shape))

    def eye(self):
        if len(self.ket_set) == 0 or len(self.bra_set) == 0:
            return (self * self.H).eye()

        ket_size = np.product([len(x.indices) for x in self.ket_set])
        bra_size = np.product([len(x.indices) for x in self.bra_set])
        if bra_size != ket_size:
            raise HilbertShapeError(bra_size, ket_size)

        return self.array(self.base_field.eye(bra_size), reshape=True)

    def basis_vec(self, idx):
        ret = self.array()
        index_map = ret._index_key_to_map(idx)
        if len(index_map) != len(self.shape):
            raise HilbertIndexError('not enough indices given')
        ret[idx] = 1
        return ret

    def assert_ket_space(self):
        if self.bra_set:
            raise NotKetSpaceError(repr(self))

class HilbertAtom(HilbertSpace):
    def __init__(self, label, latex_label, indices, base_field, dual=None):
        is_dual = not dual is None

        self.indices = indices
        self.label = label
        if latex_label is None:
            latex_label = label
        self.latex_label = latex_label
        self.is_dual = is_dual

        self._prime = None

        self._hashval = hash(self.label)
        if self.is_dual:
            self._hashval += 1

        HilbertSpace.__init__(self, None, None, base_field)

        ket_shape = [len(x.indices) for x in self.sorted_kets]
        bra_shape = [len(x.indices) for x in self.sorted_bras]
        self.shape = tuple(ket_shape + bra_shape)

        if dual:
            self._H = dual
        else:
            self._H = HilbertAtom(label, latex_label,
                indices, base_field, self)

    def __cmp__(self, other):
        if not isinstance(other, HilbertAtom):
            return HilbertSpace.__cmp__(self, other)
        else:
            if self.label < other.label:
                return -1
            elif self.label > other.label:
                return 1

            # It is not allowed for HilbertAtom's to have the same name but
            # different index set
            if self.indices != other.indices:
                raise MismatchedIndexSetError('Two instances of HilbertSpace '+
                    'with label "'+repr(self.label)+'" but with different '+
                    'indices: '+repr(self.indices)+' vs. '+repr(other.indices))

            if self.is_dual < other.is_dual:
                return -1
            elif self.is_dual > other.is_dual:
                return 1
            else:
                return 0

    def __hash__(self):
        return self._hashval

    @property
    def prime(self):
        if self._prime is None:
            if self.is_dual:
                self._prime = self.H.prime.H
            else:
                self._prime = self.base_field.indexed_space(
                    self.label+"'", self.indices, "{"+self.latex_label+"}'")
        return self._prime

    def bra(self, idx):
        if self.is_dual:
            return self.H.basis_vec({self.H: idx})
        else:
            return self.basis_vec({self: idx})

    def ket(self, idx):
        if self.is_dual:
            return self.basis_vec({self: idx})
        else:
            return self.H.basis_vec({self.H: idx})

    # These are implemented as properties rather than setting them in the
    # HilbertSpace constructor in order to avoid self-reference (which would
    # mess up pickle).

    @property
    def bra_ket_set(self):
        return frozenset([self])

    @property
    def bra_set(self):
        if self.is_dual:
            return frozenset([self])
        else:
            return frozenset()

    @property
    def ket_set(self):
        if self.is_dual:
            return frozenset()
        else:
            return frozenset([self])

    @property
    def sorted_bras(self):
        if self.is_dual:
            return [self]
        else:
            return []

    @property
    def sorted_kets(self):
        if self.is_dual:
            return []
        else:
            return [self]

    # Special operators

    def pauliX(self):
        if len(self.indices) != 2:
            raise NotImplementedError("pauliX is only implemented for qubits")
        else:
            return self.O.array([[0, 1], [1, 0]])

    def pauliY(self):
        if len(self.indices) != 2:
            raise NotImplementedError("pauliY is only implemented for qubits")
        else:
            j = self.base_field.complex_unit()
            return self.O.array([[0, -j], [j, 0]])

    def pauliZ(self, order=1):
        ret = self.O.array()
        N = len(self.indices)
        for (i, k) in enumerate(self.indices):
            ret[k, k] = self.base_field.fractional_phase(i*order, N)
        return ret

    @property
    def X(self):
        return self.pauliX()

    @property
    def Y(self):
        return self.pauliY()

    @property
    def Z(self):
        return self.pauliZ()

    def hadamard(self):
        if len(self.indices) != 2:
            raise NotImplementedError("hadamard is only implemented for qubits")
        else:
            return self.O.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def gateS(self):
        if len(self.indices) != 2:
            raise NotImplementedError("gateS is only implemented for qubits")
        else:
            j = self.base_field.complex_unit()
            return self.O.array([[1, 0], [0, j]])

    def gateT(self):
        if len(self.indices) != 2:
            raise NotImplementedError("gateT is only implemented for qubits")
        else:
            ph = self.base_field.fractional_phase(1, 8)
            return self.O.array([[1, 0], [0, ph]])

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
                if np.product(data.shape) != np.product(hs.shape):
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

################################

# FIXME - hide inside a lookup function
complex_field = HilbertBaseField(complex, 'complex_field')

# convenience functions

def indexed_space(label, indices):
    return complex_field.indexed_space(label, indices)

def qudit(label, dim):
    return complex_field.qudit(label, dim)

def qubit(label):
    return complex_field.qubit(label)
