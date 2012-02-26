"""
HilbertAtom repsenents the individible unit of HilbertSpace.  At the same time,
HilbertSpace is a base class for HilbertAtom.  These are combined into larger
product spaces by using the multiplication operator.  HilbertAtoms should be
created using the factory functions in :mod:`qitensor.factory`.
"""

import weakref
import numpy as np

from qitensor.exceptions import MismatchedIndexSetError,HilbertError
from qitensor.space import HilbertSpace

__all__ = ['HilbertAtom']

def _unreduce_v1(label, latex_label, indices, group_op, base_field, is_dual):
    """
    This is the function that handles restoring a pickle.
    """

    atom = base_field._atom_factory(label, latex_label, indices, group_op)
    return atom.H if is_dual else atom

_atom_cache = weakref.WeakValueDictionary()

def _cached_atom_factory(label, latex_label, indices, group_op, base_field):
    """This should be called only by ``qitensor.factory._atom_factory``."""

    if latex_label is None:
        latex_label = label

    indices = tuple(indices)

    key = (label, latex_label, indices, group_op, base_field)

    if not _atom_cache.has_key(key):
        atom = HilbertAtom(label, latex_label, indices, \
            group_op, base_field, None)
        _atom_cache[key] = atom
    return _atom_cache[key]

def _assert_all_compatible(collection):
    by_label = {}

    for x in collection:
        assert isinstance(x, HilbertAtom)
        if not by_label.has_key(x.label):
            by_label[x.label] = []
        by_label[x.label].append(x)

    for group in by_label.values():
        for atom in group:
            group[0]._assert_compatible(atom)

class HilbertAtom(HilbertSpace):
    def __init__(self, label, latex_label, indices, group_op, base_field, dual):
        """
        Users should not call this constructor directly, rather use the
        methods in qitensor.factory.

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a'); ha
        |a>
        >>> hb = qudit('b', 3); hb
        |b>

        sage: from qitensor import qudit
        sage: ha = qudit('a', 3); ha
        |a>
        sage: TestSuite(ha).run()
        """

        #print "init", label, dual

        assert label is not None
        assert latex_label is not None
        assert indices is not None
        assert group_op is not None

        self.label = label
        self.latex_label = latex_label
        self.indices = indices
        self.group_op = group_op

        self.is_dual = not dual is None
        self.key = (label, indices, group_op, base_field, self.is_dual)
        self._hashval = hash(self.key)
        self._prime = None

        # Many fields that HilbertSpace.__init__ would normally set are instead
        # implemented as properties (later on in this source file).
        HilbertSpace.__init__(self, None, None)

        self.base_field = base_field

        self.shape = (len(indices), )
        self._dim = len(indices)
        self._is_simple_dyad = False

        self._array_axes = [self]
        self._array_axes_lookup = {self: 0}

        if dual:
            self._H = dual
        else:
            self._H = HilbertAtom(label, latex_label,
                indices, group_op, base_field, self)

    def __reduce__(self):
        """
        Tells pickle how to store this object.
        """
        return _unreduce_v1, (self.label, self.latex_label, \
            self.indices, self.group_op, self.base_field, self.is_dual)

    def _mycmp(self, other):
        assert isinstance(other, HilbertAtom)
        return cmp(self.key, other.key)

    def _assert_compatible(self, other):
        """
        It is not allowed for HilbertAtom's with the same name but other
        properties different to be used together (leniency is given for
        latex_label).  This performs that consistency check.

        >>> from qitensor import qudit, indexed_space
        >>> qudit('a', 3)._assert_compatible(qudit('a', 3))
        >>> qudit('a', 3)._assert_compatible(qudit('a', 4))
        Traceback (most recent call last):
            ...
        MismatchedIndexSetError: 'Two instances of HilbertSpace with label "a" but with different indices: (0, 1, 2) vs. (0, 1, 2, 3)'
        >>> qudit('a', 3)._assert_compatible(indexed_space('a', (10,20,30)))
        Traceback (most recent call last):
            ...
        MismatchedIndexSetError: 'Two instances of HilbertSpace with label "a" but with different indices: (0, 1, 2) vs. (10, 20, 30)'
        >>> from qitensor.factory import GroupOpTimes_factory
        >>> qudit('a', 3)._assert_compatible(indexed_space('a', (0,1,2), group_op=GroupOpTimes_factory())) # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        MismatchedIndexSetError: 'Two instances of HilbertSpace with label "a" but with different group_op: ...

        sage: from qitensor import qudit
        sage: qudit('a', 3)._assert_compatible(qudit('a', 3, dtype=SR)) # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        MismatchedIndexSetError: 'Two instances of HilbertSpace with label "a" but with different base_field: ...
        """

        if self.indices != other.indices:
            raise MismatchedIndexSetError('Two instances of HilbertSpace '+
                'with label "'+str(self.label)+'" but with different '+
                'indices: '+repr(self.indices)+' vs. '+repr(other.indices))

        if self.group_op != other.group_op:
            raise MismatchedIndexSetError('Two instances of HilbertSpace '+
                'with label "'+str(self.label)+'" but with different '+
                'group_op: '+repr(self.group_op)+' vs. '+repr(other.group_op))

        if self.base_field != other.base_field:
            raise MismatchedIndexSetError('Two instances of HilbertSpace '+
                'with label "'+str(self.label)+'" but with different '+
                'base_field: '+repr(self.base_field)+' vs. '+
                repr(other.base_field))

    def __eq__(self, other):
        if not isinstance(other, HilbertAtom):
            return HilbertSpace.__eq__(self, other)
        else:
            return self._hashval == other._hashval and 0 == self._mycmp(other)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        """
        Compares two HilbertSpace objects lexicographically.
        """

        if not isinstance(other, HilbertAtom):
            return HilbertSpace.__lt__(self, other)
        else:
            return self._mycmp(other) < 0

    def __gt__(self, other):
        """
        Compares two HilbertSpace objects lexicographically.
        """

        if not isinstance(other, HilbertAtom):
            return HilbertSpace.__gt__(self, other)
        else:
            return self._mycmp(other) > 0

    def __ge__(self, other):
        """
        Compares two HilbertSpace objects lexicographically.
        """

        if not isinstance(other, HilbertAtom):
            return HilbertSpace.__ge__(self, other)
        else:
            return self._mycmp(other) >= 0

    def __le__(self, other):
        """
        Compares two HilbertSpace objects lexicographically.
        """

        if not isinstance(other, HilbertAtom):
            return HilbertSpace.__le__(self, other)
        else:
            return self._mycmp(other) <= 0

    def __hash__(self):
        return self._hashval

    @property
    def prime(self):
        """
        Returns a ``HilbertAtom`` just like this one but with an apostrophe
        appended to the label.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.prime
        |a'>
        >>> ha.prime.prime
        |a''>
        """
        if self._prime is None:
            if self.is_dual:
                self._prime = self.H.prime.H
            else:
                self._prime = self.base_field._atom_factory(
                    self.label+"'", "{"+self.latex_label+"}'", \
                    self.indices, self.group_op)
        return self._prime

    def ket(self, idx):
        """
        Returns a ket basis vector.

        The returned vector has a 1 in the slot corresponding to ``idx`` and
        zeros elsewhere.

        :param idx: a member of this space's index set

        >>> from qitensor import qubit, indexed_space
        >>> ha = qubit('a')
        >>> hx = indexed_space('x', ['x', 'y', 'z'])

        >>> ha.ket(0)
        HilbertArray(|a>,
        array([ 1.+0.j,  0.+0.j]))

        >>> hx.ket('y')
        HilbertArray(|x>,
        array([ 0.+0.j,  1.+0.j,  0.+0.j]))
        """

        if self.is_dual:
            return self.H.basis_vec({self.H: idx})
        else:
            return self.basis_vec({self: idx})

    def bra(self, idx):
        """
        Returns a bra basis vector.

        The returned vector has a 1 in the slot corresponding to ``idx`` and
        zeros elsewhere.

        :param idx: a member of this space's index set

        >>> from qitensor import qubit, indexed_space
        >>> ha = qubit('a')
        >>> hx = indexed_space('x', ['x', 'y', 'z'])

        >>> ha.bra(0)
        HilbertArray(<a|,
        array([ 1.+0.j,  0.+0.j]))

        >>> hx.bra('y')
        HilbertArray(<x|,
        array([ 0.+0.j,  1.+0.j,  0.+0.j]))
        """

        if self.is_dual:
            return self.basis_vec({self: idx})
        else:
            return self.H.basis_vec({self.H: idx})

    # These are implemented as properties rather than setting them in the
    # HilbertSpace constructor in order to avoid self-reference.

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

    # Special states

    def fourier_basis_state(self, k):
        """
        Returns a state from the Fourier basis.

        The returned state is :math:`\sum_j (1/\sqrt{D}) e^{2 \pi i j k/D} |j>`
        where `D` is the dimension of the space and `j` is an integer
        (regardless of the actual values of the index set).

        >>> from qitensor import qudit, indexed_space
        >>> import numpy
        >>> numpy.set_printoptions(suppress = True)

        >>> ha = qudit('a', 4)
        >>> ha.fourier_basis_state(0)
        HilbertArray(|a>,
        array([ 0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j]))
        >>> ha.fourier_basis_state(1)
        HilbertArray(|a>,
        array([ 0.5+0.j ,  0.0+0.5j, -0.5+0.j , -0.0-0.5j]))
        >>> hb = indexed_space('b', ['w', 'x', 'y', 'z'])
        >>> hb.fourier_basis_state(0)
        HilbertArray(|b>,
        array([ 0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j]))
        """

        ret = self.array()
        N = len(self.indices)
        for (i, key) in enumerate(self.indices):
            ret[key] = self.base_field.fractional_phase(i*k, N)
        return ret / self.base_field.sqrt(N)

    def x_plus(self):
        """
        Returns a state with 1/sqrt(D) in each slot.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> '%g' % ha.x_plus().norm()
        '1'
        >>> ha.X * ha.x_plus() == ha.x_plus()
        True
        """

        if len(self.indices) != 2:
            return self.fourier_basis_state(0)
        else:
            return self.array([1, 1]) / self.base_field.sqrt(2)

    def x_minus(self):
        """
        Returns the state [1, -1]/sqrt(2), only available for qubits.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> '%g' % ha.x_minus().norm()
        '1'
        >>> ha.X * ha.x_minus() == -ha.x_minus()
        True
        """

        if len(self.indices) != 2:
            raise HilbertError('x_minus only available for qubits')
        return self.array([1, -1]) / self.base_field.sqrt(2)

    def y_plus(self):
        """
        Returns the state [1, I]/sqrt(2), only available for qubits.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> '%g' % ha.y_plus().norm()
        '1'
        >>> ha.Y * ha.y_plus() == ha.y_plus()
        True
        """

        if len(self.indices) != 2:
            raise HilbertError('y_plus only available for qubits')
        i = self.base_field.complex_unit()
        return self.array([1, i]) / self.base_field.sqrt(2)

    def y_minus(self):
        """
        Returns the state [1, I]/sqrt(2), only available for qubits.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> '%g' % ha.y_minus().norm()
        '1'
        >>> ha.Y * ha.y_minus() == -ha.y_minus()
        True
        """

        if len(self.indices) != 2:
            raise HilbertError('y_minus only available for qubits')
        i = self.base_field.complex_unit()
        return self.array([1, -i]) / self.base_field.sqrt(2)

    def z_plus(self):
        """
        Returns the state [1, 0], only available for qubits.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> '%g' % ha.z_plus().norm()
        '1'
        >>> ha.Z * ha.z_plus() == ha.z_plus()
        True
        """

        if len(self.indices) != 2:
            raise HilbertError('z_plus only available for qubits')
        return self.array([1, 0])

    def z_minus(self):
        """
        Returns the state [0, 1], only available for qubits.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> '%g' % ha.z_minus().norm()
        '1'
        >>> ha.Z * ha.z_minus() == -ha.z_minus()
        True
        """

        if len(self.indices) != 2:
            raise HilbertError('z_minus only available for qubits')
        return self.array([0, 1])

    def bloch(self, theta, phi):
        """
        Returns a qubit state given its Bloch sphere representation (in radians).

        >>> import numpy as np
        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.bloch(0, 123) == ha.z_plus()
        True
        >>> ha.bloch(np.pi, 0).closeto(ha.z_minus())
        True
        >>> ha.bloch(np.pi/2, 0).closeto(ha.x_plus())
        True
        >>> ha.bloch(np.pi/2, np.pi/2).closeto(ha.y_plus())
        True
        """

        if len(self.indices) != 2:
            raise HilbertError('bloch only available for qubits')
        return self.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])

    # Special operators

    def pauliX(self, h=None, left=True):
        """
        Returns the Pauli X operator.

        If `h` is not given, then the operator [[0, 1], [1, 0]] is returned (an
        error is thrown if this is not a qubit space).

        If `h` is given, then the group Pauli X operator is returned.
        If ``left`` is True, the return value is :math:`\sum_g |g><h*g|`.
        If ``left`` is False, the return value is :math:`\sum_g |g><g*h|`.
        For qudit spaces, the default group operation is modular addition.  For
        indexed_space spaces the default operation is multiplication, and an
        error is thrown if the index set is not closed under this operation.

        See also: :func:`X`

        >>> from qitensor import qubit, qudit

        >>> ha = qubit('a')
        >>> ha.pauliX()
        HilbertArray(|a><a|,
        array([[ 0.+0.j,  1.+0.j],
               [ 1.+0.j,  0.+0.j]]))

        >>> hb = qudit('b', 3)
        >>> hb.pauliX(1)
        HilbertArray(|b><b|,
        array([[ 0.+0.j,  1.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  1.+0.j],
               [ 1.+0.j,  0.+0.j,  0.+0.j]]))
        """

        if h is None:
            if len(self.indices) != 2:
                raise NotImplementedError("h param is required except for qubits")
            else:
                return self.O.array([[0, 1], [1, 0]])
        else:
            ret = self.O.array()
            gop = self.group_op
            for g in self.indices:
                bra = gop.op(h, g) if left else gop.op(g, h)
                ret[{ self: g, self.H: bra }] = 1
            return ret

    def pauliY(self):
        """
        Returns the Pauli Y operator.

        This is only available for qubit spaces.

        See also: :func:`Y`

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.pauliY()
        HilbertArray(|a><a|,
        array([[ 0.+0.j, -0.-1.j],
               [ 0.+1.j,  0.+0.j]]))
        """

        if len(self.indices) != 2:
            raise NotImplementedError("pauliY is only implemented for qubits")
        else:
            j = self.base_field.complex_unit()
            return self.O.array([[0, -j], [j, 0]])

    def pauliZ(self, order=1):
        r"""
        Returns the generalized Pauli Z operator.

        :param order: if given, :math:`Z^\textrm{order}` will be returned.  This is
            only useful for spaces that are larger than qubits.
        :type order: integer; default 1

        See also: :func:`Z`

        >>> from qitensor import qubit, indexed_space
        >>> import numpy
        >>> numpy.set_printoptions(suppress = True)

        >>> ha = qubit('a')
        >>> ha.pauliZ()
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]]))

        >>> hx = indexed_space('x', ['x', 'y', 'z', 'w'])
        >>> hx.pauliZ()
        HilbertArray(|x><x|,
        array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j, -0.-1.j]]))

        >>> hx.pauliZ(2)
        HilbertArray(|x><x|,
        array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  1.-0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]]))
        """

        if len(self.indices) == 2 and order == 1:
            return self.O.array([[1, 0], [0, -1]])
        else:
            ret = self.O.array()
            N = len(self.indices)
            for (i, k) in enumerate(self.indices):
                ret[k, k] = self.base_field.fractional_phase(i*order, N)
            return ret

    @property
    def X(self):
        """
        Returns the Pauli X operator.

        This is only available for qubit spaces.

        See also: :func:`pauliX`

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.X
        HilbertArray(|a><a|,
        array([[ 0.+0.j,  1.+0.j],
               [ 1.+0.j,  0.+0.j]]))
        """
        return self.pauliX()

    @property
    def Y(self):
        """
        Returns the Pauli Y operator.

        This is only available for qubit spaces.

        See also: :func:`pauliY`

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.Y
        HilbertArray(|a><a|,
        array([[ 0.+0.j, -0.-1.j],
               [ 0.+1.j,  0.+0.j]]))
        """
        return self.pauliY()

    @property
    def Z(self):
        """
        Returns the generalized Pauli Z operator.

        See also: :func:`pauliZ`

        >>> from qitensor import qubit, indexed_space
        >>> import numpy
        >>> numpy.set_printoptions(suppress = True)

        >>> ha = qubit('a')
        >>> ha.Z
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]]))

        >>> hx = indexed_space('x', ['x', 'y', 'z', 'w'])
        >>> hx.Z
        HilbertArray(|x><x|,
        array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j, -0.-1.j]]))
        """
        return self.pauliZ()

    def hadamard(self):
        """
        Returns the Hadamard operator.

        This is only available for qubit spaces.

        >>> from qitensor import qubit
        >>> import numpy
        >>> numpy.set_printoptions(suppress = True, precision = 6)

        >>> ha = qubit('a')
        >>> ha.hadamard()
        HilbertArray(|a><a|,
        array([[ 0.707107+0.j,  0.707107+0.j],
               [ 0.707107+0.j, -0.707107+0.j]]))
        """

        if len(self.indices) != 2:
            raise NotImplementedError("hadamard is only implemented for qubits")
        else:
            return self.O.array([[1, 1], [1, -1]]) / self.base_field.sqrt(2)

    def gateS(self):
        """
        Returns the S-gate operator.

        This is only available for qubit spaces.

        >>> from qitensor import qubit
        >>> ha = qubit('a')
        >>> ha.gateS()
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+1.j]]))
        """

        if len(self.indices) != 2:
            raise NotImplementedError("gateS is only implemented for qubits")
        else:
            j = self.base_field.complex_unit()
            return self.O.array([[1, 0], [0, j]])

    def gateT(self):
        """
        Returns the T-gate operator.

        This is only available for qubit spaces.

        >>> from qitensor import qubit
        >>> import numpy
        >>> numpy.set_printoptions(suppress = True, precision = 6)

        >>> ha = qubit('a')
        >>> ha.gateT()
        HilbertArray(|a><a|,
        array([[ 1.000000+0.j      ,  0.000000+0.j      ],
               [ 0.000000+0.j      ,  0.707107+0.707107j]]))
        """

        if len(self.indices) != 2:
            raise NotImplementedError("gateT is only implemented for qubits")
        else:
            ph = self.base_field.fractional_phase(1, 8)
            return self.O.array([[1, 0], [0, ph]])
