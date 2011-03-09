.. qitensor documentation master file, created by
   sphinx-quickstart on Sat Mar  5 19:57:28 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qitensor's documentation!
====================================

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

Contents:

.. toctree::
   :maxdepth: 2

   examples/index
   reference
   todo

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

