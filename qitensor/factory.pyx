"""
This module contains functions for creating HilbertSpace's and
HilbertBaseField's.  This is the preferred entry point for using the qitensor
package.
"""

from qitensor import have_sage
from qitensor.basefield import HilbertBaseField
import qitensor.atom
cimport qitensor.atom

__all__ = ['base_field_lookup', 'indexed_space', 'qubit', 'qudit', 'GroupOpCyclic_factory', 'GroupOpTimes_factory']

##############################

cdef class GroupOpCyclic_impl(object):
    cdef long D

    def __init__(self, D):
        """Don't use this constructor, rather call ``GroupOpCyclic_factory``."""
        self.D = D

    def __reduce__(self):
        """
        Tells pickle how to store this object.
        """
        return GroupOpCyclic_factory, (self.D,)

    def op(self, x, y):
        """
        The group operation (modular addition).

        >>> from qitensor import GroupOpCyclic_factory
        >>> g = GroupOpCyclic_factory(5)
        >>> g.op(2, 7)
        4
        """
        return (x+y) % self.D

# This implements memoization
cdef dict _op_cyclic_cache = {}
cpdef GroupOpCyclic_factory(D):
    """
    Returns an instance of GroupOpCyclic_impl, the cyclic group of order D.

    >>> from qitensor import GroupOpCyclic_factory
    >>> g = GroupOpCyclic_factory(5)
    >>> g.op(2, 7)
    4
    """
    if not _op_cyclic_cache.has_key(D):
        _op_cyclic_cache[D] = GroupOpCyclic_impl(D)
    return _op_cyclic_cache[D]

##############################

cdef class GroupOpTimes_impl(object):
    def __init__(self):
        """Don't use this constructor, rather call ``GroupOpTimes_factory``."""
        pass

    def __reduce__(self):
        """
        Tells pickle how to store this object.
        """
        return GroupOpTimes_factory, tuple()

    def op(self, x, y):
        """
        The group operation (multiplication).

        >>> from qitensor import GroupOpTimes_factory
        >>> g = GroupOpTimes_factory()
        >>> g.op(3, 4)
        12
        """
        return x*y

# This implements memoization
cdef GroupOpTimes_impl _op_times_cache = GroupOpTimes_impl()
cpdef GroupOpTimes_factory():
    """
    Returns an instance of GroupOpTimes_impl, which operates on elements by multiplication.

    >>> from qitensor import GroupOpTimes_factory
    >>> g = GroupOpTimes_factory()
    >>> g.op(3, 4)
    12
    """
    return _op_times_cache

##############################

cdef list _base_field_factories = []

import qitensor.basefield
_base_field_factories.append(qitensor.basefield._factory)

if have_sage:
    import qitensor.sagebasefield
    _base_field_factories.append(qitensor.sagebasefield._factory)
    
cpdef base_field_lookup(dtype):
    r"""
    Returns the HilbertBaseField for the given data type.

    :param dtype: the base field data type
    :type dtype: python type or Sage CommutativeRing or HilbertBaseField;
        default complex

    >>> from qitensor import base_field_lookup
    >>> base_field_lookup(complex).__class__
    <type 'qitensor.basefield.HilbertBaseField'>
    """

    if isinstance(dtype, HilbertBaseField):
        return dtype

    for f in _base_field_factories:
        ret = f(dtype)
        if ret is not None:
            return ret

    raise NotImplementedError("data type not supported")

##############################

cpdef indexed_space(label, indices, dtype=complex, latex_label=None, group_op=None):
    r"""
    Returns a finite-dimensional Hilbert space with an arbitrary index set.

    :param label: a unique label for this Hilbert space
    :param indices: a sequence defining the index set
    :param dtype: the base field data type
    :type dtype: python type or Sage CommutativeRing or HilbertBaseField;
        default complex
    :param latex_label: an optional latex representation of the label
    :param group_op: group operation

    ``group_op``, if given, should be a class that defines an
    ``op(self, x, y)`` method.  This supports things like the generalized
    pauliX operator.  The default is ``op(self, x, y) = x*y``.  The
    ``qubit`` and ``qudit`` constructors use ``op(self, x, y) = (x+y)%D``.

    This is really just a shortcut for 
        ``base_field_lookup(dtype).indexed_space( ... )``

    See also: :func:`qubit`, :func:`qudit`

    >>> from qitensor import indexed_space
    >>> ha = indexed_space('a', ['x', 'y', 'z'])
    >>> ha
    |a>
    >>> ha.indices
    ('x', 'y', 'z')
    """

    field = base_field_lookup(dtype)

    if group_op is None:
        group_op = GroupOpTimes_factory()

    return qitensor.atom._atom_factory(field, label, latex_label, indices, group_op)

cpdef qudit(label, dim, dtype=complex, latex_label=None):
    r"""
    Returns a finite-dimensional Hilbert space with index set [0, 1, ..., n-1].

    :param label: a unique label for this Hilbert space
    :param dim: the dimension of the Hilbert space
    :param dtype: the base field data type
    :type dtype: python type or Sage CommutativeRing or HilbertBaseField;
        default complex
    :param latex_label: an optional latex representation of the label

    This is really just a shortcut for 
        ``base_field_lookup(dtype).qudit( ... )``

    See also: :func:`qubit`, :func:`indexed_space`

    >>> from qitensor import qudit
    >>> ha = qudit('a', 3)
    >>> ha
    |a>
    >>> ha.indices
    (0, 1, 2)
    """

    group_op = GroupOpCyclic_factory(dim)

    return indexed_space(
        label=label, indices=range(dim), dtype=dtype,
        latex_label=latex_label, group_op=group_op)

cpdef qubit(label, dtype=complex, latex_label=None):
    r"""
    Returns a two-dimensional Hilbert space with index set [0, 1].

    :param label: a unique label for this Hilbert space
    :param dtype: the base field data type
    :type dtype: python type or Sage CommutativeRing or HilbertBaseField;
        default complex
    :param latex_label: an optional latex representation of the label

    This is really just a shortcut for 
        ``base_field_lookup(dtype).qubit( ... )``

    See also: :func:`qudit`, :func:`indexed_space`

    >>> from qitensor import qubit
    >>> ha = qubit('a')
    >>> ha
    |a>
    >>> ha.indices
    (0, 1)
    """

    return qudit(label=label, dim=2, dtype=dtype, latex_label=latex_label)
