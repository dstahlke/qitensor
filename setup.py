#!/usr/bin/env python

from distutils.core import setup
from distutils.core import Command
import unittest

import qitensor
import qitensor.tests.hilbert

# Adapted from sympy
class test_qitensor(Command):
    """Runs all tests under the qitensor/ folder
    """

    description = "Automatically run the test suite for qitensor."
    user_options = []  # distutils complains if this is not here.

    def __init__(self, *args):
        self.args = args[0] # so we can pass it to other classes
        Command.__init__(self, *args)

    def initialize_options(self):  # distutils wants this
        pass

    def finalize_options(self):    # this too
        pass

    def run(self):
        suite = unittest.TestSuite([
            qitensor.tests.hilbert.suite(),
        ])
        unittest.TextTestRunner().run(suite)

setup(
    name = 'qitensor',
    version = qitensor.__version__,
    author = 'Dan Stahlke',
    author_email = 'dstahlke@gmail.com',
    url = 'http://www.stahlke.org/dan/qitensor',
    license = 'BSD',
    keywords = ['quantum', 'tensor', 'numpy'],
    description = 'Quantum Tensors',
    long_description = '''
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
    ''',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        ],
    packages = [
        'qitensor',
        'qitensor.tests',
    ],
    cmdclass = {'test': test_qitensor },
)
