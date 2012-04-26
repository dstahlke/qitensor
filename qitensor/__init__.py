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

__version__ = "0.10dev"

########################################

try:
    import sage.all
    have_sage = True
except ImportError:
    have_sage = False

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

def doctest():
    """Runs all doctests and unit tests."""

    import doctest
    import qitensor.tests.hilbert
    import qitensor.tests.experimental

    # Only works under Sage:
    #qitensor.sagebasefield
    # Benchmarks: no applicable doctests
    #qitensor.benchmark_cy, \
    #qitensor.benchmark_py, \
    #qitensor.tests.bench, \
    # These are unit tests, they don't contain doctests
    #qitensor.tests.experimental, \
    #qitensor.tests.hilbert:

    doctest_modules = [ \
        qitensor.array, \
        qitensor.arrayformatter, \
        qitensor.atom, \
        qitensor.basefield, \
        qitensor.circuit, \
        qitensor.exceptions, \
        qitensor.factory, \
        qitensor.space, \
        qitensor.subspace, \
        qitensor.experimental.cartan_decompose, \
        qitensor.experimental.cartan_decompose_impl, \
        qitensor.experimental.stabilizers, \
    ]

    print "\nRunning doctests..."
    for m in doctest_modules:
        print m.__name__, ('.'*(45-len(m.__name__))), doctest.testmod(m)

    # FIXME - these tests are probably obsoleted by the doctests.  It might not
    # be worth keeping them around.
    print "\nRunning unit tests..."
    import unittest
    suite = unittest.TestSuite([
        qitensor.tests.hilbert.suite(),
        qitensor.tests.experimental.suite(),
    ])
    unittest.TextTestRunner().run(suite)
