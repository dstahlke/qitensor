"""
HilbertAtom repsenents the individible unit of HilbertSpace (i.e.
qubits or qudits).  At the same time, HilbertSpace is a base class for
HilbertAtom.  These are combined into larger product spaces by using the
multiplication operator.  HilbertAtoms should be created using the factory
functions in :mod:`qitensor.factory`.
"""

#import weakref
import numpy as np

import qitensor
from qitensor.exceptions import MismatchedSpaceError,HilbertError
from qitensor.space import HilbertSpace
from qitensor.space cimport HilbertSpace

__all__ = ['HilbertAtom', 'direct_sum']

def _unreduce_v1(label, latex_label, indices, group_op, base_field, is_dual, addends=None):
    """
    This is the function that handles restoring a pickle.
    """

    atom = _atom_factory(base_field, label, latex_label, indices, group_op)
    atom.addends = addends
    if atom.addends is not None:
        atom._create_addend_isoms()
    return atom.H if is_dual else atom

#_atom_cache = weakref.WeakValueDictionary()
cdef dict _atom_cache = dict()

cpdef _atom_factory(base_field, label, latex_label, indices, group_op):
    r"""
    Factory method for creating ``HilbertAtom`` objects.

    Subclasses can override this method in order to return custom
    subclasses of ``HilbertAtom``.

    Users should call methods in ``qitensor.factory`` instead.
    """

    assert label is not None
    assert indices is not None
    assert group_op is not None

    if latex_label is None:
        latex_label = label

    # convert to proper types
    cdef str _label = str(label)
    cdef str _latex_label = str(latex_label)
    cdef tuple _indices = tuple(indices)

    cdef tuple key = (_label, _latex_label, _indices, group_op, base_field)

    if not key in _atom_cache:
        atom = HilbertAtom(_label, _latex_label, _indices, \
            group_op, base_field, None)
        _atom_cache[key] = atom
    return _atom_cache[key]

cpdef _assert_all_compatible(collection):
    """
    Make sure that all HilbertAtoms in a collection that have the same label
    are compatible.

    >>> from qitensor import qudit
    >>> from qitensor.atom import _assert_all_compatible
    >>> _assert_all_compatible([qudit('a', 3), qudit('a', 3)])
    >>> _assert_all_compatible([qudit('a', 3), qudit('b', 4)])
    >>> _assert_all_compatible([qudit('a', 3), qudit('a', 4)])
    Traceback (most recent call last):
        ...
    MismatchedSpaceError: 'Two instances of HilbertSpace with label "a" but with different indices: (0, 1, 2) vs. (0, 1, 2, 3)'
    """

    by_label = {}

    for x in collection:
        assert isinstance(x, HilbertAtom)
        if not x.label in by_label:
            by_label[x.label] = []
        by_label[x.label].append(x)

    for group in by_label.values():
        for atom in group:
            group[0]._assert_compatible(atom)

cdef class HilbertAtom(HilbertSpace):
    def __init__(self, str label, str latex_label, tuple indices, group_op, base_field, dual):
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

        # (Sphinx docstrings)
        #: The text label for this atom (gets displayed as ``|label>`` or ``<label|``).
        self.label = label
        #: The label used for the latex representation (by default equal to ``self.label``).
        self.latex_label = latex_label
        #: A tuple of the tokens used as indices for this space.  By default this consists of
        #: the integers ``0..dim-1``.
        self.indices = indices
        #: The group operation associated with the indices.  This is relevant to the
        #: ``self.pauliX`` operator.  Typically this is modular addition.
        self.group_op = group_op

        #: True if this atom is a bra space.
        self.is_dual = not dual is None
        #: The unique key used for comparing this atom to other atoms.
        self.key = (label, indices, group_op, base_field, self.is_dual)
        self._hashval = hash(self.key)

        #: The HilbertBaseField that defines the numerical properties of arrays belonging
        #: to this space.
        self.base_field = base_field

        if dual:
            _H = dual
        else:
            _H = HilbertAtom(label, latex_label,
                indices, group_op, base_field, self)

        # NOTE: since 'self' is passed in the bra_set/ket_set parameters to the
        # superclass constructor, it is necessary that some of the properties
        # are set before the superclass constructor is called (this is done
        # above).

        if self.is_dual:
            HilbertSpace.__init__(<HilbertSpace>self, frozenset(), frozenset([self]), _H)
        else:
            HilbertSpace.__init__(<HilbertSpace>self, frozenset([self]), frozenset(), _H)

    def __reduce__(self):
        """
        Tells pickle how to store this object.
        """
        return _unreduce_v1, (self.label, self.latex_label, \
            self.indices, self.group_op, self.base_field, self.is_dual, self.addends)

    cpdef _mycmp(self, other):
        """
        Helper function used by __lt__, __gt__, __eq__, etc.
        """

        assert isinstance(other, HilbertAtom)
        #return cmp(self.key, other.key)
        return (self.key > other.key) - (self.key < other.key)

    cpdef _assert_compatible(self, HilbertAtom other):
        """
        It is not allowed for HilbertAtom's with the same name but other
        properties different to be used together (leniency is given for
        latex_label).  This performs that consistency check.

        >>> from qitensor import qudit, indexed_space
        >>> qudit('a', 3)._assert_compatible(qudit('a', 3))
        >>> qudit('a', 3)._assert_compatible(qudit('a', 4))
        Traceback (most recent call last):
            ...
        MismatchedSpaceError: 'Two instances of HilbertSpace with label "a" but with different indices: (0, 1, 2) vs. (0, 1, 2, 3)'
        >>> qudit('a', 3)._assert_compatible(indexed_space('a', (10,20,30)))
        Traceback (most recent call last):
            ...
        MismatchedSpaceError: 'Two instances of HilbertSpace with label "a" but with different indices: (0, 1, 2) vs. (10, 20, 30)'
        >>> from qitensor.factory import GroupOpTimes_factory
        >>> qudit('a', 3)._assert_compatible(indexed_space('a', (0,1,2), group_op=GroupOpTimes_factory())) # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        MismatchedSpaceError: 'Two instances of HilbertSpace with label "a" but with different group_op: ...

        sage: from qitensor import qudit
        sage: qudit('a', 3)._assert_compatible(qudit('a', 3, dtype=SR)) # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        MismatchedSpaceError: 'Two instances of HilbertSpace with label "a" but with different base_field: ...
        """

        if self.indices != other.indices:
            raise MismatchedSpaceError('Two instances of HilbertSpace '+
                'with label "'+str(self.label)+'" but with different '+
                'indices: '+repr(self.indices)+' vs. '+repr(other.indices))

        if self.group_op != other.group_op:
            raise MismatchedSpaceError('Two instances of HilbertSpace '+
                'with label "'+str(self.label)+'" but with different '+
                'group_op: '+repr(self.group_op)+' vs. '+repr(other.group_op))

        if self.base_field != other.base_field:
            raise MismatchedSpaceError('Two instances of HilbertSpace '+
                'with label "'+str(self.label)+'" but with different '+
                'base_field: '+repr(self.base_field)+' vs. '+
                repr(other.base_field))

    def __richcmp__(self, other, op):
        """
        Rich comparison of two HilbertSpace objects.

        >>> from qitensor import qubit, qudit
        >>> ha = qubit('a')
        >>> hb = qudit('b', 3)
        >>> (ha == ha, ha == hb)
        (True, False)
        >>> (ha != ha, ha != hb)
        (False, True)
        >>> (ha < ha, ha < hb, hb < ha)
        (False, True, False)
        >>> (ha > ha, ha > hb, hb > ha)
        (False, False, True)
        >>> (ha >= ha, ha >= hb, hb >= ha)
        (True, False, True)
        >>> (ha <= ha, ha <= hb, hb <= ha)
        (True, True, False)
        """

        if not isinstance(other, HilbertAtom):
            if op == 0: # <
                return HilbertSpace.__gt__(self, other)
            elif op == 1: # <=
                return HilbertSpace.__ge__(self, other)
            elif op == 2: # ==
                return HilbertSpace.__eq__(self, other)
            elif op == 3: # !=
                return HilbertSpace.__ne__(self, other)
            elif op == 4: # >
                return HilbertSpace.__lt__(self, other)
            elif op == 5: # >=
                return HilbertSpace.__le__(self, other)

        if op == 0: # <
            return self._mycmp(other) < 0
        elif op == 1: # <=
            return self._mycmp(other) <= 0
        elif op == 2 or op == 3: # == or !=
            eq = self._hashval == other._hashval and 0 == self._mycmp(other)
            return eq if op==2 else not eq
        elif op == 4: # >
            return self._mycmp(other) > 0
        elif op == 5: # >=
            return self._mycmp(other) >= 0

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
                self._prime = _atom_factory(
                    self.base_field,
                    self.label+"'", "{"+self.latex_label+"}'",
                    self.indices, self.group_op)
        return self._prime

    cpdef ket(self, idx):
        """
        Returns a ket basis vector.

        The returned vector has a 1 in the slot corresponding to ``idx`` and
        zeros elsewhere.

        :param idx: a member of this space's index set

        >>> from qitensor import qubit, indexed_space
        >>> ha = qubit('a')
        >>> hb = indexed_space('b', ['x', 'y', 'z'])

        >>> ha.ket(0)
        HilbertArray(|a>,
        array([ 1.+0.j,  0.+0.j]))

        >>> hb.ket('y')
        HilbertArray(|b>,
        array([ 0.+0.j,  1.+0.j,  0.+0.j]))
        """

        if self.is_dual:
            return self.H.basis_vec({self.H: idx})
        else:
            return self.basis_vec({self: idx})

    cpdef bra(self, idx):
        """
        Returns a bra basis vector.

        The returned vector has a 1 in the slot corresponding to ``idx`` and
        zeros elsewhere.

        :param idx: a member of this space's index set

        >>> from qitensor import qubit, indexed_space
        >>> ha = qubit('a')
        >>> hb = indexed_space('b', ['x', 'y', 'z'])

        >>> ha.bra(0)
        HilbertArray(<a|,
        array([ 1.+0.j,  0.+0.j]))

        >>> hb.bra('y')
        HilbertArray(<b|,
        array([ 0.+0.j,  1.+0.j,  0.+0.j]))
        """

        if self.is_dual:
            return self.basis_vec({self: idx})
        else:
            return self.H.basis_vec({self.H: idx})

    # Special states

    cpdef x_plus(self):
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

    cpdef x_minus(self):
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

    cpdef y_plus(self):
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

    cpdef y_minus(self):
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

    cpdef z_plus(self):
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

    cpdef z_minus(self):
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

    cpdef bloch(self, theta, phi):
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

    cpdef pauliX(self, h=None, left=True):
        """
        Returns the Pauli X operator.

        If `h` is given, then the group Pauli X operator is returned.
        If ``left`` is True, the return value is :math:`\sum_g |g><h*g|`.
        If ``left`` is False, the return value is :math:`\sum_g |g><g*h|`.
        For qudit spaces, the default group operation is modular addition.  For
        indexed_space spaces the default operation is multiplication, and an
        error is thrown if the index set is not closed under this operation.

        If `h` is not given, the default of `1` is used for cyclic addition
        groups (the default group), otherwise an error is thrown.

        NOTE: some people use a convention that is a transpose of this!

        See also: :func:`X`

        >>> import numpy as np
        >>> from qitensor import qubit, qudit, indexed_space, dihedral_group

        >>> ha = qubit('a')
        >>> ha.pauliX()
        HilbertArray(|a><a|,
        array([[ 0.+0.j,  1.+0.j],
               [ 1.+0.j,  0.+0.j]]))

        >>> hb = qudit('b', 3)
        >>> hb.pauliX()
        HilbertArray(|b><b|,
        array([[ 0.+0.j,  1.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  1.+0.j],
               [ 1.+0.j,  0.+0.j,  0.+0.j]]))

        >>> hb.pauliX(2)
        HilbertArray(|b><b|,
        array([[ 0.+0.j,  0.+0.j,  1.+0.j],
               [ 1.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  1.+0.j,  0.+0.j]]))

        >>> S3 = dihedral_group(3)
        >>> hc = indexed_space('c', S3.elements)
        >>> np.all([hc.pauliX(f) * hc.ket(f*g) == hc.ket(g) for f in S3.elements for g in S3.elements])
        True
        >>> np.all([hc.pauliX(f, left=False) * hc.ket(g*f) == hc.ket(g) for f in S3.elements for g in S3.elements])
        True
        """

        if h is None:
            if len(self.indices) == 2:
                return self.O.array([[0, 1], [1, 0]])
            if self.group_op.__class__ == qitensor.factory.GroupOpCyclic_impl:
                h = 1
            else:
                raise NotImplementedError("h param is required for pauliX when groups are used")

        ret = self.O.array()
        gop = self.group_op
        for g in self.indices:
            bra = gop.op(h, g) if left else gop.op(g, h)
            ret[{ self: g, self.H: bra }] = 1
        return ret

    cpdef pauliY(self):
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

    cpdef pauliZ(self, int order=1):
        r"""
        Returns the generalized Pauli Z operator.

        :param order: if given, :math:`Z^\textrm{order}` will be returned.  This is
            only useful for spaces that are larger than qubits.
        :type order: integer; default 1

        The return value is :math:`\sum_k e^{2 \pi i k / d} |k><k|`.

        See also: :func:`Z`

        >>> from qitensor import qubit, indexed_space

        >>> ha = qubit('a')
        >>> ha.pauliZ()
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]]))

        >>> hb = indexed_space('b', ['x', 'y', 'z', 'w'])
        >>> hb.pauliZ()
        HilbertArray(|b><b|,
        array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j, -0.-1.j]]))

        >>> hb.pauliZ(2)
        HilbertArray(|b><b|,
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

        >>> ha = qubit('a')
        >>> ha.Z
        HilbertArray(|a><a|,
        array([[ 1.+0.j,  0.+0.j],
               [ 0.+0.j, -1.+0.j]]))

        >>> hb = indexed_space('b', ['x', 'y', 'z', 'w'])
        >>> hb.Z
        HilbertArray(|b><b|,
        array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j, -0.-1.j]]))
        """
        return self.pauliZ()

    cpdef gateS(self):
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

    cpdef gateT(self):
        """
        Returns the T-gate operator.

        This is only available for qubit spaces.

        >>> from qitensor import qubit

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

    @classmethod
    def direct_sum(cls, kets):
        """
        Returns the direct sum of this atom with another, along with a pair
        of isometries mapping to the sum space.

        >>> from qitensor import qudit, direct_sum
        >>> ha = qudit('a', 2)
        >>> hb = qudit('b', 3)
        >>> hab = direct_sum((ha, hb))
        >>> (hab, hab.P[0].space, hab.P[1].space)
        (|a+b>, |a+b><a|, |a+b><b|)
        >>> x = ha.random_array()
        >>> y = hb.random_array()
        >>> z = hab.P[0]*x + hab.P[1]*y
        >>> x == hab.P[0].H * z
        True
        >>> y == hab.P[1].H * z
        True

        >>> # it is allowed to repeat a space
        >>> haa = direct_sum((ha, ha))
        >>> (haa, haa.P[0].space, haa.P[1].space)
        (|a+a>, |a+a><a|, |a+a><a|)
        >>> x1 = ha.random_array()
        >>> x2 = ha.random_array()
        >>> z = haa.P[0]*x1 + haa.P[1]*x2
        >>> x1 == haa.P[0].H * z
        True
        >>> x2 == haa.P[1].H * z
        True
        """

        assert len(kets) > 1
        for k in kets:
            if not isinstance(k, HilbertAtom):
                raise TypeError('direct_sum only applies to HilbertAtoms')
            k.base_field.assert_same(kets[0].base_field)

        label = '+'.join([ k.label if k.addends is None else '('+k.label+')' for k in kets ])
        latex_label = ' \oplus '.join([ k.latex_label if k.addends is None else '('+k.latex_label+')' for k in kets ])
        dim = np.sum([ k.dim() for k in kets ])
        ket_sum = qitensor.qudit(label, dim, dtype=kets[0].base_field, latex_label=latex_label)
        ket_sum.addends = kets
        ket_sum._create_addend_isoms()

        return ket_sum

    cpdef _create_addend_isoms(self):
        self.P = []
        idx = 0
        for k in self.addends:
            isom = (self*k.H).array()
            isom.nparray[idx:idx+k.dim(), :] = np.eye(k.dim())
            self.P.append(isom)
            idx += k.dim()

direct_sum = HilbertAtom.direct_sum
