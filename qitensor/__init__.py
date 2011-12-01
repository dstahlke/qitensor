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

########################################

try:
    import sage.all
    have_sage = True
except ImportError:
    have_sage = False

########################################

def shape_product(l):
    return np.prod(l, dtype=int)

########################################

class PrintOptions: pass

PRINT_OPTS = PrintOptions()
PRINT_OPTS.precision = 6
PRINT_OPTS.suppress = True
PRINT_OPTS.suppress_thresh = 1e-12
# FIXME - option for html vs. latex vs. none for ipython pretty printing

def set_printoptions(precision=None, suppress=None, suppress_thresh=None):
    if precision is not None:
        PRINT_OPTS.precision = precision
    if suppress is not None:
        PRINT_OPTS.suppress = suppress
    if suppress_thresh is not None:
        PRINT_OPTS.suppress_thresh = suppress_thresh

########################################

from qitensor.exceptions import *
from qitensor.basefield import *
from qitensor.space import *
from qitensor.atom import *
from qitensor.array import *
from qitensor.factory import *
from qitensor.circuit import *
import qitensor.experimental

__all__ = \
    exceptions.__all__ + \
    basefield.__all__ + \
    space.__all__ + \
    atom.__all__ + \
    array.__all__ + \
    factory.__all__ + \
    circuit.__all__ + \
    ['set_printoptions']
