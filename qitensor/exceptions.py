"""
Various exceptions that can be raised by qitensor functions.
"""

__all__ = [
    'DuplicatedSpaceError',
    'HilbertError',
    'HilbertIndexError',
    'HilbertShapeError',
    'MismatchedSpaceError',
    'NotKetSpaceError',
]

class HilbertError(Exception):
    """
    The generic exception used by qitensor.  All of this package's other
    exceptions derive from this one.
    """

    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

class MismatchedSpaceError(HilbertError):
    """
    Raised when two HilbertAtoms have the same label but different properties,
    or when the HilbertSpace requested for an operation doesn't match that of
    an array.
    """

    def __init__(self, msg):
        HilbertError.__init__(self, msg)

class DuplicatedSpaceError(HilbertError):
    """
    Raised when an operation would result in an output with two copies of a
    HilbertAtom.

    >>> from qitensor import qubit
    >>> ha = qubit('a')
    >>> ha * ha
    Traceback (most recent call last):
        ...
    DuplicatedSpaceError: '|a>'
    >>> x = ha.array()
    >>> x * x
    Traceback (most recent call last):
        ...
    DuplicatedSpaceError: '|a>'
    """

    def __init__(self, spaces, msg=None):
        if msg is None:
            msg = repr(spaces)
        else:
            msg = msg+': '+repr(spaces)
        HilbertError.__init__(self, msg)

class HilbertIndexError(HilbertError, LookupError):
    """
    Raised when an invalid array subscript is given.

    >>> from qitensor import qubit
    >>> ha = qubit('a')
    >>> x = ha.array()
    >>> x[2]
    Traceback (most recent call last):
        ...
    HilbertIndexError: 'Index set for |a> does not contain 2'
    >>> x[{ ha.H: 0 }]
    Traceback (most recent call last):
        ...
    HilbertIndexError: 'Hilbert space not part of this array: <a|'
    """

    def __init__(self, msg):
        HilbertError.__init__(self, msg)
        LookupError.__init__(self, )

class HilbertShapeError(HilbertError, ValueError):
    """
    Raised when an array is not the right shape for the requested operation.
    """

    def __init__(self, shape1, shape2):
        msg = repr(shape1)+' vs. '+repr(shape2)
        HilbertError.__init__(self, msg)
        ValueError.__init__(self, )

class NotKetSpaceError(HilbertError):
    """
    Raised when a bra space is given to an operation that only works on ket spaces.
    """

    def __init__(self, msg):
        HilbertError.__init__(self, msg)
