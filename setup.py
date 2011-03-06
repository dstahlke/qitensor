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
    description = 'Quantum Tensors',
    author = 'Dan Stahlke',
    author_email = 'dstahlke@gmail.com',
    url = 'http://www.stahlke.org/dan/qitensor',
    license = 'BSD',
    packages = [
        'qitensor',
        'qitensor.tests',
    ],
    cmdclass = {'test': test_qitensor },
)
