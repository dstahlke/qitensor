#!/usr/bin/env python

from setuptools import setup
from setuptools import Command
from setuptools.extension import Extension
import re
import sys
import sysconfig
import os

# http://stackoverflow.com/a/4515279/1048959
try:
    import Cython
    from Cython.Distutils import build_ext
    # Lower versions give code that causes segfault.  Version 0.16 and above is
    # tested to work.
    use_cython = (Cython.__version__ >= '0.16')
except Exception:
    use_cython = False

version = [m.group(1) for m in [re.search('__version__ = "(.*)"', line) for line in open('qitensor/__init__.py').readlines()] if m is not None][0]
# Cannot import qitensor until cython extensions are built
# import qitensor
# version = qitensor.__version__

cmdclass = { }

######################################################################

# http://stackoverflow.com/a/14369968/1048959
def build_dir_name(dname):
    """Returns the name of the build directory"""
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)

# Adapted from sympy
class test_qitensor(Command):
    """Runs tests."""

    description = "Automatically run the test suite for qitensor."
    user_options = []  # setuptools complains if this is not here.

    def __init__(self, *args):
        self.args = args[0] # so we can pass it to other classes
        Command.__init__(self, *args)

    def initialize_options(self):  # setuptools wants this
        pass

    def finalize_options(self):    # this too
        pass

    def run(self):
        buildpath = os.path.join('build', build_dir_name('lib'))
        if not os.path.isfile(os.path.join(buildpath, 'qitensor', '__init__.py')):
            print('Error: you must do "./setup.py build" first.')
            sys.exit(1)
        sys.path.insert(0, buildpath)

        import numpy
        numpy.set_printoptions(precision=6, suppress=True)

        import qitensor
        qitensor.doctest()

cmdclass.update({'test': test_qitensor})

######################################################################

ext_names = [
    "array",
    "atom",
    "basefield",
    "benchmark_cy",
    "factory",
    "sagebasefield",
    "sympybasefield",
    "space"
]

ext_modules = []

if use_cython:
    cmdclass.update({'build_ext': build_ext})

    ext_modules += [
        Extension("qitensor."+s, ["qitensor/"+s+".pyx"])
        for s in ext_names
    ]

    for e in ext_modules:
        # renamed from pyrex_directives in Cython 0.16.
        e.cython_directives = {
            "embedsignature": True,
            "nonecheck": True,
            #"profile": True,
        }
        e.depends = [
            "qitensor/array.pxd",
            "qitensor/atom.pxd",
            "qitensor/basefield.pxd",
            "qitensor/factory.pxd",
            "qitensor/space.pxd",
        ]
else: # no cython
    ext_modules += [
        Extension("qitensor."+s, ["qitensor/"+s+".c"])
        for s in ext_names
    ]

# This doesn't seem to make it faster:
#for e in ext_modules:
#    e.extra_compile_args = ['-O2']

######################################################################

setup(
    name='qitensor',
    version=version,
    author='Dan Stahlke',
    author_email='dstahlke@gmail.com',
    url='http://www.stahlke.org/dan/qitensor',
    license='BSD',
    keywords=['quantum', 'tensor', 'numpy', 'sage'],
    description='Quantum Hilbert Space Tensors in Python and Sage',
    long_description='''
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
    packages=[
        'qitensor',
        'qitensor.experimental',
        'qitensor.tests',
    ],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=['cvxopt', 'numpy', 'scipy'],
)
