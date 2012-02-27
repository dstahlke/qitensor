"""
Various exceptions that can be raised by qitensor functions.
"""

__all__ = [
    'DuplicatedSpaceError',
    'HilbertError',
    'HilbertIndexError',
    'HilbertShapeError',
    'IncompatibleBaseFieldError',
    'MismatchedIndexSetError',
    'NotKetSpaceError',
    'HilbertSliceError',
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

class MismatchedIndexSetError(HilbertError):
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

class HilbertSliceError(HilbertError):
    def __init__(self, msg):
        HilbertError.__init__(self, msg)
