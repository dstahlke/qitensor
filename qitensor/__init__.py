r"""
Quantum Hilbert Space Tensors in Python and Sage

This module is essentially a wrapper for numpy that uses semantics useful for
finite dimensional quantum mechanics of many particles.  In particular, this
should be useful for the study of quantum information and quantum computing.
Each array is associated with a tensor-product Hilbert space.  The underlying
spaces can be bra spaces or ket spaces and are indexed using any finite
sequence (typically a range of integers starting from zero, but any sequence is
allowed).  When arrays are multiplied, a tensor contraction is performed among
the bra spaces of the left array and the ket spaces of the right array.
Various linear algebra methods are available which are aware of the Hilbert
space tensor product structure.

* Component Hilbert spaces have string labels (e.g. ``qubit('a') * qubit('b')``
  gives ``|a,b>``).
* Component spaces are finite dimensional and are indexed either by integers or
  by any sequence (e.g. elements of a group).
* In Sage, it is possible to create arrays over the Symbolic Ring.
* Multiplication of arrays automatically contracts over the intersection of the
  bra space of the left factor and the ket space of the right factor.
* Linear algebra routines such as SVD are provided which are aware of the
  Hilbert space labels.
"""

__version__ = "0.8.1"

import numpy as np
import operator

########################################

try:
    import sage.all
    have_sage = True
except ImportError:
    have_sage = False

########################################

def _shape_product(l):
    """
    Multiplies a tuple of integers together.

    Used to convert shape to dimension.

    >>> from qitensor import _shape_product
    >>> _shape_product((1,2,3,4))
    24
    """

    # faster than np.prod(l, dtype=int)
    return reduce(operator.mul, l, 1)

########################################

from qitensor.exceptions import *
from qitensor.basefield import *
from qitensor.space import *
from qitensor.atom import *
from qitensor.array import *
from qitensor.factory import *
from qitensor.circuit import *
import qitensor.experimental
from qitensor.arrayformatter import *
from qitensor.subspace import *

__all__ = \
    qitensor.exceptions.__all__ + \
    qitensor.basefield.__all__ + \
    qitensor.space.__all__ + \
    qitensor.atom.__all__ + \
    qitensor.array.__all__ + \
    qitensor.factory.__all__ + \
    qitensor.circuit.__all__ + \
    qitensor.arrayformatter.__all__ + \
    qitensor.subspace.__all__
