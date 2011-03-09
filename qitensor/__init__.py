r"""
Quantum Hilbert Space Tensors

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

AUTHORS:
- Dan Stahlke (2011-03-05): initial version

"""

__version__ = "0.1"

from exceptions import *
from basefield import *
from space import *
from atom import *
from array import *
from functions import *

__all__ = \
    exceptions.__all__ + \
    basefield.__all__ + \
    space.__all__ + \
    atom.__all__ + \
    array.__all__ + \
    functions.__all__
