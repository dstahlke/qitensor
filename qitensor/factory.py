"""
This module contains functions for creating HilbertSpace's and
HilbertBaseField's.  This is the preferred entry point for using the qitensor
package.
"""

from qitensor import have_sage

__all__ = ['base_field_lookup', 'indexed_space', 'qubit', 'qudit']

###########################

_base_field_factories = []

import qitensor.basefield
_base_field_factories.append(qitensor.basefield.factory)

if have_sage:
    import qitensor.sagebasefield
    _base_field_factories.append(qitensor.sagebasefield.factory)
    
###########################

def base_field_lookup(dtype):
    r"""
    Returns the HilbertBaseField for the given data type.

    :param dtype: the base field data type
    :type dtype: python type or Sage CommutativeRing; default complex

    >>> from qitensor import base_field_lookup
    >>> base_field_lookup(complex).__class__
    <class 'qitensor.basefield.HilbertBaseField'>
    """

    for f in _base_field_factories:
        ret = f(dtype)
        if ret is not None:
            return ret

    raise NotImplementedError("data type not supported")

def indexed_space(label, indices, dtype=complex, latex_label=None, group_op=None):
    r"""
    Returns a finite-dimensional Hilbert space with an arbitrary index set.

    :param label: a unique label for this Hilbert space
    :param indices: a sequence defining the index set
    :param dtype: the base field data type
    :type dtype: python type or Sage CommutativeRing; default complex
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
    return field.indexed_space(label, indices, \
        latex_label=latex_label, group_op=group_op)

def qudit(label, dim, dtype=complex, latex_label=None):
    r"""
    Returns a finite-dimensional Hilbert space with index set [0, 1, ..., n-1].

    :param label: a unique label for this Hilbert space
    :param dim: the dimension of the Hilbert space
    :param dtype: the base field data type
    :type dtype: python type or Sage CommutativeRing; default complex
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

    field = base_field_lookup(dtype)
    return field.qudit(label, dim, latex_label)

def qubit(label, dtype=complex, latex_label=None):
    r"""
    Returns a two-dimensional Hilbert space with index set [0, 1].

    :param label: a unique label for this Hilbert space
    :param dtype: the base field data type
    :type dtype: python type or Sage CommutativeRing; default complex
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

    field = base_field_lookup(dtype)
    return field.qubit(label, latex_label)
