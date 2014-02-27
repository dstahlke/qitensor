#!/usr/bin/python

import numpy as np
import numpy.linalg as linalg

from qitensor.array import HilbertArray
from qitensor.space import HilbertSpace
from qitensor.exceptions import MismatchedSpaceError

# This is the only thing that is exported.
__all__ = ['TensorSubspace']

def _unreduce_v1(basis, perp_basis, tol, hilb_space, dtype):
    """
    This is the function that handles restoring a pickle.
    """

    return TensorSubspace(basis, perp_basis, tol, hilb_space, dtype)

class TensorSubspace(object):
    """
    Represents a subspace of vectors, matrices, or tensors.

    Methods are available for projecting to this subspace, computing span and intersection of
    subspaces, etc.  This module can be used independently of the rest of qitensor, and the
    doctest below reflects this.  However, if you are using qitensor then typically a subspace
    would be created using one of the following methods:

    * :func:`qitensor.array.HilbertArray.span`
    * :func:`qitensor.space.HilbertSpace.full_space`
    * :func:`qitensor.space.HilbertSpace.empty_space`

    >>> import numpy
    >>> from qitensor import TensorSubspace
    >>> x = TensorSubspace.from_span(numpy.random.randn(4,5,10))
    >>> x
    <TensorSubspace of dim 4 over space (5, 10)>
    >>> x.dim()
    4
    >>> # TensorSubspace can act like a list of orthonormal basis vectors.
    >>> x[0] in x
    True
    >>> x.perp()[0] in x
    False
    >>> y = TensorSubspace.from_span(numpy.random.randn(30,5,10))
    >>> y
    <TensorSubspace of dim 30 over space (5, 10)>
    >>> z = TensorSubspace.from_span(numpy.random.randn(30,5,10))
    >>> z
    <TensorSubspace of dim 30 over space (5, 10)>
    >>> x | y # span of two subspaces
    <TensorSubspace of dim 34 over space (5, 10)>
    >>> y | z
    <TensorSubspace of dim 50 over space (5, 10)>
    >>> y & z # intersection of two subspaces
    <TensorSubspace of dim 10 over space (5, 10)>
    >>> # `<` and `>` check whether one subspace contains the other
    >>> y > x&y
    True
    >>> y > x|y
    False
    >>> x|y > y
    True
    >>> y - x # subspace of y perpendicular to x
    <TensorSubspace of dim 26 over space (5, 10)>
    >>> # `~x` gives the perpendicular subspace, equivalent to x.perp().
    >>> (y & z).equiv(~(~y | ~z))
    True
    >>> (y-(y-x)).equiv(TensorSubspace.from_span([ y.project(v) for v in x ]))
    True
    """

    def __init__(self, basis, perp_basis, tol, hilb_space, dtype, validate=False):
        """
        Don't directly instantiate this class using this constructor, use the
        ``TensorSubspace.from_span`` factory method instead.

        :param basis: an orthonormal basis for this subspace
        :type basis: list of numpy arrays or ``HilbertArray``s, or numpy array whose first axis
            indexes the operators
        :param perp_basis: an orthonormal basis for the perpendicular subspace
        :param tol: tolerance to use for checking whether vectors are perpendicular
        :param hilb_space: the ``HilbertSpace`` that this is a subspace of, if any
        :param dtype: the numpy dtype to use
        :param validate: if true, a check is made that the arguments indeed represent an
            orthonormal basis
        """

        assert dtype is not None

        # Convert input to an nparray of numbers (as opposed to a list of nparrays or a list of
        # HilbertArrays.
        def to_nparray(l):
            return np.array([ x.nparray if isinstance(x, HilbertArray) else x
                    for x in l ], dtype=dtype)
        basis = to_nparray(basis)
        perp_basis = to_nparray(perp_basis)

        # basis and perp_basis must be arrays of the appropriate shape, even if they are empty.
        # If one of them is empty, then copy the shape from the other one.  The empty basis
        # will be a numpy array of shape (0, ...).
        if basis.shape[0] == 0:
            basis = np.zeros(((0,)+perp_basis.shape[1:]), basis.dtype)
        if perp_basis.shape[0] == 0:
            perp_basis = np.zeros(((0,)+basis.shape[1:]), basis.dtype)

        self._tol = tol
        self._hilb_space = hilb_space
        self._dtype = dtype
        self._basis = basis
        self._perp_basis = perp_basis
        self._dim = basis.shape[0]
        self._col_shp = basis.shape[1:]
        self._col_dim = np.product(self._col_shp)
        self._perp_cache = None
        self._hermit_cache = None
        # can be passed to constructor to make a space with similar configuration
        self._config_kw = { 'tol': tol, 'hilb_space': hilb_space, 'dtype': dtype }

        if hilb_space is not None:
            assert isinstance(hilb_space, HilbertSpace)
            assert hilb_space.shape == basis.shape[1:]
            assert dtype == hilb_space.base_field.dtype

        assert basis.shape[1:] == perp_basis.shape[1:]
        assert self._dim + perp_basis.shape[0] == self._col_dim

        # These contain a copy of the basis and perpendicular basis in which each basis element
        # has been flattened into a vector.
        self._basis_flat = basis.reshape((self._dim, self._col_dim))
        self._perp_basis_flat = perp_basis.reshape(((self._col_dim-self._dim), self._col_dim))

        if validate:
            products = np.tensordot(self._basis_flat.conjugate(), self._basis_flat, axes=((1,),(1,)))
            assert linalg.norm(products - np.eye(self._dim)) < self._tol

            products = np.tensordot(self._perp_basis_flat.conjugate(), self._perp_basis_flat, axes=((1,),(1,)))
            assert linalg.norm(products - np.eye(self._col_dim - self._dim)) < self._tol

            products = np.tensordot(self._basis_flat.conjugate(), self._perp_basis_flat, axes=((1,),(1,)))
            assert linalg.norm(products) < self._tol

    def __reduce__(self):
        """
        Tells pickle how to store this object.

        >>> from qitensor import qubit, indexed_space
        >>> import pickle
        >>> ha = qubit('a')
        >>> hb = indexed_space('b', ['x', 'y', 'z'])
        >>> x = (ha*hb).random_array()
        >>> S = x.span(hb); S
        <TensorSubspace of dim 2 over space (|b>)>
        >>> T = pickle.loads(pickle.dumps(S)); T
        <TensorSubspace of dim 2 over space (|b>)>
        >>> S.equiv(T)
        True
        """

        return _unreduce_v1, (
            self._basis,
            self._perp_basis,
            self._tol,
            self._hilb_space,
            self._dtype,
            )

    @classmethod
    def from_span(cls, X, tol=1e-10, hilb_space=None, dtype=None):
        """
        Construct a ``TensorSubspace`` that represents the span of the given operators.

        :param X: the operators to take the span of.
        :type X: list of numpy arrays or HilbertArray, or numpy array whose first axis indexes the operators
        :param tol: tolerance for determining whether operators are perpendicular
        :param dtype: the datatype (default is to use the datatype of the input operators)

        >>> from qitensor import TensorSubspace
        >>> import numpy as np
        >>> x = np.random.randn(3, 5)
        >>> y = np.random.randn(3, 5)
        >>> TensorSubspace.from_span([x, y])
        <TensorSubspace of dim 2 over space (3, 5)>
        >>> TensorSubspace.from_span([x, y, 2*x+0.3*y])
        <TensorSubspace of dim 2 over space (3, 5)>
        >>> TensorSubspace.from_span(x)
        <TensorSubspace of dim 3 over space (5,)>
        >>> [0,1,-1] in TensorSubspace.from_span([[1,1,0], [1,0,1]])
        True
        >>> [0,1,1]  in TensorSubspace.from_span([[1,1,0], [1,0,1]])
        False
        """

        # The input must either be a numpy array or be convertible to a list.
        if not isinstance(X, np.ndarray):
            X = list(X)

        # If the basis elements are of type HilbertArray, then do some validation and convert
        # them to numpy arrays.
        if isinstance(X[0], HilbertArray):
            assert hilb_space is None or hilb_space == X[0].space
            hilb_space = X[0].space
            for op in X:
                assert isinstance(op, HilbertArray)
                assert op.space == hilb_space
            X = [op.nparray for op in X]

        if hilb_space is not None:
            if dtype is None:
                dtype = hilb_space.base_field.dtype
            else:
                assert dtype == hilb_space.base_field.dtype

        # Convert the basis now to a numpy array.  Let numpy choose a dtype if none was
        # specified.
        X = np.array(X, dtype=dtype)
        if dtype is None:
            dtype = X.dtype

        # Minimum type is float (i.e. avoid integer types)
        dtype = np.find_common_type([dtype], [np.float32])
        X = X.astype(dtype)

        assert len(X.shape) >= 2

        # First axis of X enumerates the basis elements, remaining axes belong to the basis
        # elements themselves.
        col_shp = X.shape[1:]
        num_bases = X.shape[0]
        element_dimension = np.product(col_shp)

        if hilb_space is not None:
            assert col_shp == hilb_space.shape

        if num_bases==0:
            if hilb_space is not None:
                col_shp = hilb_space
            return cls.empty(col_shp, tol=tol)

        # Now convert X to a rank 2 tensor.  Each basis element is treated as a vector.
        X = X.reshape(num_bases, element_dimension)

        # Form an orthonormal basis from the basis X.  The QR transform method is faster, but
        # seems to have problems sometimes.  The SVD method seems to work well.
        use_qr = False
        if use_qr:
            # I think that sometimes QR doesn't do exactly what I want.  The code
            # below potentially includes extra vectors into the basis.
            assert 0
            # Append random values to that Y represents the whole space.
            # This allows us to also generate the basis orthogonal to span{X}.
            Y = np.concatenate((X, np.random.rand(element_dimension, element_dimension)), axis=0)
            (Q, R) = linalg.qr(Y.transpose(), mode='full')
            dim = np.sum([linalg.norm(row) > tol for row in R[:, :num_bases]])
            basis      = Q[:, :dim].transpose().reshape((dim,)+col_shp)
            perp_basis = Q[:, dim:].transpose().reshape((element_dimension-dim,)+col_shp)
        else:
            (_U, s, V) = linalg.svd(X, full_matrices=True)
            dim = np.sum(s > tol)
            basis      = V[:dim, :].reshape((dim,)+col_shp)
            perp_basis = V[dim:, :].reshape((element_dimension-dim,)+col_shp)

        return cls(basis, perp_basis, tol=tol, hilb_space=hilb_space, dtype=dtype)

    @classmethod
    def empty(cls, col_shp, tol=1e-10, dtype=complex):
        """
        Constructs the empty subspace of the given dimension.

        >>> from qitensor import TensorSubspace, qubit
        >>> TensorSubspace.empty((3,5))
        <TensorSubspace of dim 0 over space (3, 5)>
        >>> ha = qubit('a')
        >>> TensorSubspace.empty(ha)
        <TensorSubspace of dim 0 over space (|a>)>
        """

        if isinstance(col_shp, HilbertSpace):
            hilb_space = col_shp
            dtype = hilb_space.base_field.dtype
            col_shp = col_shp.shape
        else:
            hilb_space = None

        n = np.product(col_shp)
        basis = np.zeros((0,)+col_shp, dtype=dtype)
        perp_basis = np.eye(n).reshape((n,)+col_shp)
        return cls(basis, perp_basis, tol=tol, hilb_space=hilb_space, dtype=dtype)

    @classmethod
    def full(cls, col_shp, tol=1e-10, dtype=complex):
        """
        Constructs the full subspace of the given dimension.

        >>> from qitensor import TensorSubspace, qubit
        >>> TensorSubspace.full((3,5))
        <TensorSubspace of dim 15 over space (3, 5)>
        >>> ha = qubit('a')
        >>> TensorSubspace.full(ha)
        <TensorSubspace of dim 2 over space (|a>)>
        """

        return cls.empty(col_shp, tol=tol, dtype=dtype).perp()

    @classmethod
    def create_random_hermitian(cls, spc, opspc_dim, tracefree=False):
        """
        Create an operator subspace :math:`S \subseteq \mathcal{L}(A)` such that :math:`S =
        S^\dagger, I \perp S`.  The argument ``spc`` can either be an integer (the dimension of
        ``A``) or a ``HilbertSpace``.  The argument ``opspc_dim`` sets the dimension of ``S``.
        The ``tracefree`` parameter allows creation of a trace-free space.

        >>> from qitensor import TensorSubspace, qudit
        >>> S = TensorSubspace.create_random_hermitian(3, 4)
        >>> S
        <TensorSubspace of dim 4 over space (3, 3)>
        >>> S.is_hermitian()
        True
        >>> np.eye(3) in S.perp() # we didn't specify tracefree
        False
        >>> ha = qudit('a', 4)
        >>> T = TensorSubspace.create_random_hermitian(ha, 2, tracefree=True)
        >>> T
        <TensorSubspace of dim 2 over space (|a><a|)>
        >>> T.is_hermitian()
        True
        >>> ha.eye() in T.perp()
        True
        """

        if isinstance(spc, HilbertSpace):
            spc.assert_ket_space()
            n = spc.dim()
            S = TensorSubspace.empty(spc.O)
        else:
            assert isinstance(spc, int)
            n = spc
            spc = None
            S = TensorSubspace.empty((n, n))

        ops = []
        for i in range(opspc_dim):
            M = np.random.standard_normal(size=(n,n)) + \
                np.random.standard_normal(size=(n,n))*1j
            M += M.conj().T
            if tracefree:
                M -= np.eye(n) * np.trace(M) / n
            ops.append(M)
        S = TensorSubspace.from_span(ops)

        if spc is not None:
            S = S.map(lambda x: spc.O.array(x, reshape=True))

        assert S.dim() == opspc_dim

        return S

    def basis(self):
        """
        Returns an orthonormal basis for this subspace.

        You can also just directly treat a TensorSubspace object as a list, to the same effect.
        """

        return list(self)

    def assert_compatible(self, other):
        """
        Raise error if ``other`` is not compatible with this space.  Compatible means
        that the dimensions are the same and that the HilbertSpaces match if both
        subspaces have that property.

        >>> from qitensor import TensorSubspace
        >>> spc1 = TensorSubspace.full((3,5))
        >>> spc2 = TensorSubspace.empty((3,5))
        >>> spc3 = TensorSubspace.empty((3,6))
        >>> spc1.assert_compatible(spc2)
        >>> spc1.assert_compatible(spc3)
        Traceback (most recent call last):
            ...
        AssertionError
        """

        if not isinstance(other, TensorSubspace):
            raise TypeError('other object is not a TensorSubspace')

        if self._hilb_space is not None and other._hilb_space is not None:
            assert self._hilb_space == other._hilb_space

        assert self._col_shp == other._col_shp

    def perp(self):
        """
        Returns orthogonal complement of this space.
        Equivalent to ``~self``.

        >>> from qitensor import TensorSubspace
        >>> import numpy as np
        >>> x = np.random.randn(3, 5)
        >>> y = np.random.randn(3, 5)
        >>> spc = TensorSubspace.from_span([x, y]); spc
        <TensorSubspace of dim 2 over space (3, 5)>
        >>> spc.perp()
        <TensorSubspace of dim 13 over space (3, 5)>
        >>> spc.equiv(spc.perp().perp())
        True
        >>> TensorSubspace.full((3,5)).perp().equiv(TensorSubspace.empty((3,5)))
        True
        >>> TensorSubspace.empty((3,5)).perp().equiv(TensorSubspace.full((3,5)))
        True
        """

        if self._perp_cache is None:
            self._perp_cache = TensorSubspace(self._perp_basis, self._basis, **self._config_kw)
            self._perp_cache._perp_cache = self
        return self._perp_cache

    def _str_inner(self):
        if self._hilb_space is None:
            spc_str = str(self._col_shp)
        else:
            spc_str = '('+repr(self._hilb_space)+')'
        return "dim "+str(self._dim)+" over space "+spc_str

    def __str__(self):
        """
        Returns string representation.

        >>> from qitensor import TensorSubspace
        >>> spc = TensorSubspace.full((3, 5))
        >>> str(spc)
        '<TensorSubspace of dim 15 over space (3, 5)>'
        """

        return "<TensorSubspace of "+self._str_inner()+">"

    def __repr__(self):
        """
        Returns string representation.

        >>> from qitensor import TensorSubspace
        >>> spc = TensorSubspace.full((3, 5))
        >>> str(spc)
        '<TensorSubspace of dim 15 over space (3, 5)>'
        """

        return str(self)

    def __invert__(self):
        """
        Returns orthogonal complement of this space.

        >>> from qitensor import TensorSubspace
        >>> import numpy as np
        >>> x = np.random.randn(3, 5)
        >>> y = np.random.randn(3, 5)
        >>> spc = TensorSubspace.from_span([x, y]); spc
        <TensorSubspace of dim 2 over space (3, 5)>
        >>> ~spc
        <TensorSubspace of dim 13 over space (3, 5)>
        >>> spc.equiv(~(~spc))
        True
        >>> (~TensorSubspace.full((3,5))).equiv(TensorSubspace.empty((3,5)))
        True
        >>> (~TensorSubspace.empty((3,5))).equiv(TensorSubspace.full((3,5)))
        True
        """

        return self.perp()

    def __or__(self, other):
        """
        Span of union of spaces.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> y = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> z = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> x | y
        <TensorSubspace of dim 34 over space (5, 10)>
        >>> y | z
        <TensorSubspace of dim 50 over space (5, 10)>
        >>> y > x|y
        False
        >>> x|y > y
        True
        >>> (y & z).equiv(~(~y | ~z))
        True
        """

        self.assert_compatible(other)
        b_cat = np.concatenate((self._basis, other._basis), axis=0)
        return self.from_span(b_cat, **self._config_kw)

    def __and__(self, other):
        """
        Intersection of spaces.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> y = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> z = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> y & z
        <TensorSubspace of dim 10 over space (5, 10)>
        >>> y > x&y
        True
        >>> (y & z).equiv(~(~y | ~z))
        True
        """

        return (self.perp() | other.perp()).perp()

    def __sub__(self, other):
        """
        Subspace of first space perpendicular to second space.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> y = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> z = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> y - x
        <TensorSubspace of dim 26 over space (5, 10)>
        >>> (y-(y-x)).equiv(TensorSubspace.from_span([ y.project(v) for v in x ]))
        True
        """

        return self & other.perp()

    def __mul__(self, other):
        """
        Returns span{ x*other : x in self }.

        >>> from qitensor import qudit
        >>> ha = qudit('a', 3)
        >>> S = TensorSubspace.from_span([ ha.O.random_array(), ha.O.random_array() ])
        >>> S
        <TensorSubspace of dim 2 over space (|a><a|)>
        >>> T = TensorSubspace.from_span([ ha.O.random_array(), ha.O.random_array() ])
        >>> T
        <TensorSubspace of dim 2 over space (|a><a|)>
        >>> U1 = S * T; U1
        <TensorSubspace of dim 4 over space (|a><a|)>
        >>> U2 = TensorSubspace.from_span([ x*y for x in S for y in T ])
        >>> U1.equiv(U2)
        True
        >>> (S * -2).equiv(S)
        True
        >>> rho = ha.O.random_array()
        >>> (S * rho).equiv( TensorSubspace.from_span([ x*rho for x in S ]) )
        True
        """

        if isinstance(other, TensorSubspace):
            return TensorSubspace.from_span([ x*y for x in self for y in other ])
        else:
            return TensorSubspace.from_span([ x*other for x in self ])

    def __rmul__(self, other):
        """
        Returns span{ other*x : x in self }.

        >>> from qitensor import qudit
        >>> ha = qudit('a', 3)
        >>> S = TensorSubspace.from_span([ ha.O.random_array(), ha.O.random_array() ])
        >>> S
        <TensorSubspace of dim 2 over space (|a><a|)>
        >>> (-2 * S).equiv(S)
        True
        >>> rho = ha.O.random_array()
        >>> (rho * S).equiv( TensorSubspace.from_span([ rho*x for x in S ]) )
        True
        """

        return TensorSubspace.from_span([ other*x for x in self ])

    @property
    def H(self):
        """
        Returns span{ x.H : x in self }

        >>> from qitensor import qudit
        >>> ha = qudit('a', 3)
        >>> hb = qudit('b', 4)
        >>> rho = (ha*hb.H).random_array()
        >>> sigma = (ha*hb.H).random_array()
        >>> S = TensorSubspace.from_span([ rho, sigma ])
        >>> S
        <TensorSubspace of dim 2 over space (|a><b|)>
        >>> S.H
        <TensorSubspace of dim 2 over space (|b><a|)>
        >>> S.H.equiv( TensorSubspace.from_span([ rho.H, sigma.H ]) )
        True
        """

        if self._hilb_space is not None:
            return TensorSubspace.from_span([ x.H for x in self ])
        else:
            return TensorSubspace.from_span([ x.conj().T for x in self ])

    def to_basis(self, x):
        """
        Returns a representation of tensor ``x`` as a vector in the basis of
        this subspace.  If ``x`` is not in this subspace, the orthogonal
        projection is used (i.e. the element of the subspace closest to ``x``).

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> spc = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> spc
        <TensorSubspace of dim 4 over space (5, 10)>
        >>> spc.dim()
        4
        >>> np.allclose(spc.to_basis(spc[0]), np.array([1,0,0,0]))
        True
        >>> np.allclose(spc.to_basis(spc[1]), np.array([0,1,0,0]))
        True
        >>> v = np.random.randn(5,10)
        >>> spc.to_basis(v).shape
        (4,)
        >>> w = spc.from_basis(spc.to_basis(v))
        >>> np.allclose(w, spc.project(v))
        True
        """

        if self._hilb_space is not None:
            if isinstance(x, HilbertArray):
                assert x.space == self._hilb_space
                return self.to_basis(x.nparray)

        x = np.array(x)
        assert x.shape == self._col_shp
        nd = len(x.shape)
        return np.tensordot(self._basis.conjugate(), x, axes=(list(range(1, nd+1)), list(range(nd))))

    def from_basis(self, v):
        """
        Returns the element of this subspace corresponding to the given vector,
        which is to be expressed in this subspace's basis.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> spc = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> spc
        <TensorSubspace of dim 4 over space (5, 10)>
        >>> spc.dim()
        4
        >>> np.allclose(spc[0], spc.from_basis([1,0,0,0]))
        True
        >>> np.allclose(spc[1], spc.from_basis([0,1,0,0]))
        True
        >>> v = np.random.randn(5,10)
        >>> w = spc.from_basis(spc.to_basis(v))
        >>> np.allclose(w, spc.project(v))
        True
        """

        v = np.array(v)
        assert len(v.shape) == 1
        assert v.shape[0] == self._dim
        ret = np.tensordot(v, self._basis, axes=((0,),(0,)))
        if self._hilb_space is None:
            return ret
        else:
            return self._hilb_space.array(ret)

    def project(self, x):
        """
        Returns the element of this subspace that is the closest to the given
        tensor.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> spc = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> v = np.random.randn(5,10)
        >>> np.allclose(spc[0], spc.project(spc[0]))
        True
        >>> np.allclose(v, spc.project(v))
        False
        >>> w = spc.from_basis(spc.to_basis(v))
        >>> np.allclose(w, spc.project(v))
        True
        """

        return self.from_basis(self.to_basis(x))

    def is_perp(self, other):
        """
        Tests whether the given TensorSubspace or vector is perpendicular to this space.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> y = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> x.is_perp(y)
        False
        >>> x.is_perp(~x)
        True
        >>> x.is_perp(y-x)
        True
        >>> y.is_perp(x[0])
        False
        >>> (y-x).is_perp(x[0])
        True
        """
        if isinstance(other, TensorSubspace):
            self.assert_compatible(other)
            products = np.tensordot(self._basis_flat.conjugate(), other._basis_flat, axes=((1,),(1,)))
            return linalg.norm(products) < self._tol
        else:
            return linalg.norm(self.to_basis(other)) < self._tol

    def contains(self, other):
        """
        Tests whether the given TensorSubspace or vector is contained in this space.
        Equivalent to ``self > other``.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> y = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> y.contains(x)
        False
        >>> (y|x).contains(x)
        True
        >>> x.contains(y[0])
        False
        >>> y.contains(y[0])
        True
        """

        return self.perp().is_perp(other)

    def equiv(self, other):
        """
        Tests whether this subspace is equal to ``other``, to within an error tolerance.
        Equivalent to ``self < other and self > other``.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> y = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> x.equiv(y)
        False
        >>> (x & y).equiv(~(~x | ~y))
        True
        """

        return self.contains(other) and other.contains(self)

    # Helper for is_hermitian and hermitian_basis.
    def _op_flatten(self):
        if self._hilb_space:
            return self._nomath_map(lambda x: x.as_np_matrix())
        else:
            nd = len(self._col_shp)
            assert nd % 2 == 0
            shp = self._col_shp[:nd//2]
            assert shp == self._col_shp[nd//2:]
            shp = np.product(shp)
            return self.reshape((shp, shp))

    # Helper for is_hermitian and hermitian_basis.
    def _op_unflatten(self, other):
        other = [ x.reshape(self._col_shp) for x in other ]
        if self._hilb_space:
            other = [ self._hilb_space.array(x) for x in other ]
        return other

    def is_hermitian(self):
        r"""
        A subspace S is Hermitian if :math:`x \in S \iff x^\dagger \in S`.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> S = TensorSubspace.from_span(np.random.randn(4,10,10))
        >>> S.is_hermitian()
        False
        >>> T = S | TensorSubspace.from_span([ x.transpose().conjugate() for x in S ])
        >>> T.is_hermitian()
        True
        """

        nd = len(self._col_shp)
        if nd % 2 != 0:
            return False
        shp = self._col_shp[:nd//2]
        if shp != self._col_shp[nd//2:]:
            return False

        if self._hilb_space or len(self._col_shp) > 2:
            return self._op_flatten().is_hermitian()

        assert len(self._col_shp) == 2
        assert self._col_shp[0] == self._col_shp[1]
        for x in self._basis:
            if not self.contains(x.conjugate().transpose()):
                return False
        return True

    def hermitian_basis(self):
        """
        Compute a basis consisting of Hermitian operators.  This is only allowed for Hermitian
        subspaces (see ``is_hermitian``).  This basis can be used to map real vectors to
        complex operators.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> S = TensorSubspace.from_span(np.random.randn(4,10,10))
        >>> S.hermitian_basis() # S is not Hermitian
        Traceback (most recent call last):
            ...
        AssertionError
        >>> T = S | TensorSubspace.from_span([ x.transpose().conjugate() for x in S ])
        >>> hbas = T.hermitian_basis()
        >>> hbas.shape[0] == T.dim()
        True
        >>> x = np.tensordot(np.random.randn(hbas.shape[0]), hbas, axes=([0],[0]))
        >>> np.allclose(x, x.transpose().conjugate())
        True
        >>> x in T
        True
        """

        if self._hermit_cache is not None:
            return self._hermit_cache

        if self._hilb_space or len(self._col_shp) > 2:
            hb = self._op_flatten().hermitian_basis()
            return self._op_unflatten(hb)

        assert self.is_hermitian()

        assert len(self._col_shp) == 2
        assert self._col_shp[0] == self._col_shp[1]
        n = self._col_shp[0]

        # x_to_S = [ |a>, <a| ; Si ]
        x_to_S = np.zeros((n,n, self._dim), dtype=complex)
        for (S_i, Sv) in enumerate(self._basis):
            x_to_S[:, :, S_i] = Sv
        # project onto Hermitian space while simulating complex values with
        # reals on the x side
        x_to_S_1  = x_to_S    + np.transpose(x_to_S,    [1, 0, 2]).conjugate()
        x_to_S_1j = x_to_S*1j + np.transpose(x_to_S*1j, [1, 0, 2]).conjugate()
        x_to_S = np.concatenate((x_to_S_1, x_to_S_1j), axis=2)

        # decrease parameters by only taking linearly independent subspace
        sqrmat = np.array([x_to_S.real, x_to_S.imag]).reshape(2*(n**2), 2*self._dim)
        (U, s, _V) = linalg.svd(sqrmat)
        n_indep = np.sum(s > self._tol)
        S_basis = U[:, :n_indep]
        x_to_S_reduced_real = S_basis.reshape(2,n,n, n_indep)
        x_to_S_reduced = x_to_S_reduced_real[0] + 1j*x_to_S_reduced_real[1]

        for i in range(n_indep):
            s = x_to_S_reduced[:, :, i]
            sH = s.transpose().conjugate()
            assert linalg.norm(s - sH) < 1e-13
            # correct for numerical error and make it exactly Hermitian
            x_to_S_reduced[:, :, i] = (s + sH)/2

        hbasis = x_to_S_reduced.transpose([2, 0, 1])

        if self._hilb_space is None:
            self._hermit_cache = hbasis
        else:
            self._hermit_cache = [self._hilb_space.array(x) for x in hbasis]

        return self._hermit_cache

    def tensor_prod(self, other):
        """
        >>> from qitensor import qudit

        >>> ha = qudit('a', 3)
        >>> hb = qudit('b', 4)
        >>> Sn = TensorSubspace.from_span(np.random.randn(2,3,4)); Sn
        <TensorSubspace of dim 2 over space (3, 4)>
        >>> Tn = TensorSubspace.from_span(np.random.randn(3,4,3)); Tn
        <TensorSubspace of dim 3 over space (4, 3)>
        >>> STn = Sn.tensor_prod(Tn); STn
        <TensorSubspace of dim 6 over space (3, 4, 4, 3)>
        >>> Sh = Sn.map(lambda x: (ha*hb.H).array(x)); Sh
        <TensorSubspace of dim 2 over space (|a><b|)>
        >>> Th = Tn.map(lambda x: (hb*ha.H).array(x)); Th
        <TensorSubspace of dim 3 over space (|b><a|)>
        >>> Sh.tensor_prod(Tn)
        Traceback (most recent call last):
            ...
        MismatchedSpaceError: "one factor had HilbertSpace, the other didn't"
        >>> STh = Sh.tensor_prod(Th); STh
        <TensorSubspace of dim 6 over space (|a,b><a,b|)>
        >>> # transpose maps |a,b><a,b| to |a>,<b|,|b>,<a|
        >>> STn.equiv(STh.map(lambda x: x.nparray.transpose(0,3,1,2)))
        True
        """

        if (self._hilb_space is None) != (other._hilb_space is None):
            raise MismatchedSpaceError('one factor had HilbertSpace, the other didn\'t')

        if self._hilb_space is not None:
            h1 = self._hilb_space
            h2 = other._hilb_space
            if not h1.bra_ket_set.isdisjoint(h2.bra_ket_set):
                raise MismatchedSpaceError('spaces are not disjoint')
            b_b   = [ x.tensordot(y, frozenset()) for x in  self for y in  other ]
            bp_b  = [ x.tensordot(y, frozenset()) for x in ~self for y in  other ]
            b_bp  = [ x.tensordot(y, frozenset()) for x in  self for y in ~other ]
            bp_bp = [ x.tensordot(y, frozenset()) for x in ~self for y in ~other ]
            b_b_p = np.concatenate((bp_b, b_bp, bp_bp), axis=0)
            cfg = self._config_kw.copy()
            cfg['hilb_space'] = h1*h2
            return TensorSubspace(b_b, b_b_p, **cfg)

        n = len(self._basis.shape)
        m = len(other._basis.shape)

        # Compute tensor product and shuffle axes.
        def tp(a, b):
            products = np.tensordot(a, b, axes=([], []))
            products = products.transpose([n] + list(range(n)) + list(range(n+1, n+m)))
            products = products.reshape((products.shape[0]*products.shape[1],) + products.shape[2:])
            return products

        b_b = tp(self._basis, other._basis)
        # this is simpler, but computing perp_basis manually avoids an svd/qr call
        #return TensorSubspace.from_span(b_b, **self._config_kw)
        bp_b  = tp(self._perp_basis, other._basis)
        b_bp  = tp(self._basis,  other._perp_basis)
        bp_bp = tp(self._perp_basis, other._perp_basis)
        b_b_p = np.concatenate((bp_b, b_bp, bp_bp), axis=0)
        return TensorSubspace(b_b, b_b_p, **self._config_kw)

    def map(self, f):
        r"""
        Returns span{ f(x) : x \in S }.
        """

        if self.dim() == 0:
            return self

        b_new = [ f(m) for m in self ]

        cfg = self._config_kw.copy()
        if isinstance(b_new[0], HilbertArray):
            cfg['hilb_space'] = b_new[0].space
            cfg['dtype'] = b_new[0].space.base_field.dtype
        else:
            cfg['hilb_space'] = None
            cfg['dtype'] = np.array(b_new).dtype

        return TensorSubspace.from_span(b_new, **cfg)

    def _nomath_map(self, f):
        """
        Like map, but assumes the operation preserves orthogonality.
        """

        b_new  = np.array([f(m) for m in self ])
        bp_new = np.array([f(m) for m in self.perp() ])

        element = b_new[0] if len(b_new) else bp_new[0]
        cfg = self._config_kw.copy()
        if isinstance(element, HilbertArray):
            cfg['hilb_space'] = element.space
            cfg['dtype'] = element.space.base_field.dtype
        else:
            cfg['hilb_space'] = None
            cfg['dtype'] = np.array(b_new).dtype

        return TensorSubspace(b_new, bp_new, **cfg)

    def transpose(self, axes):
        if self._hilb_space is not None:
            raise NotImplementedError()
        else:
            return self._nomath_map(lambda m: m.transpose(axes))

    def reshape(self, shape):
        if self._hilb_space is not None:
            raise NotImplementedError()
        else:
            return self._nomath_map(lambda m: m.reshape(shape))

    def random_vec(self):
        """
        Returns a random vector in this subspace.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> x
        <TensorSubspace of dim 4 over space (5, 10)>
        >>> v = x.random_vec()
        >>> v in x
        True
        >>> abs(1 - np.linalg.norm(v)) < 1e-13
        True
        >>> x.equiv(TensorSubspace.from_span([ x.random_vec() for i in range(x.dim()) ]))
        True
        """

        if self._basis.dtype.kind == 'c':
            v = np.random.randn(self._dim) + 1j*np.random.randn(self._dim)
        else:
            # if it is not complex or float, then how to make a random number?
            assert self._basis.dtype.kind == 'f'
            v = np.random.randn(self._dim)
        v /= linalg.norm(v)
        return self.from_basis(v)

    def random_hermit(self):
        """
        Returns a random Hermitian vector in this subspace.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> S = TensorSubspace.from_span(np.random.randn(4,10,10))
        >>> S.random_hermit() # S is not Hermitian
        Traceback (most recent call last):
            ...
        AssertionError
        >>> T = S | TensorSubspace.from_span([ x.transpose().conjugate() for x in S ])
        >>> v = T.random_hermit()
        >>> np.allclose(v, v.transpose().conjugate())
        True
        >>> v in T
        True
        """

        hb = self.hermitian_basis()
        v = np.random.randn(len(hb))
        v /= linalg.norm(v)
        if self._hilb_space is not None:
            return np.dot(v, hb)
        else:
            return np.tensordot(v, hb, axes=([0],[0]))

    def dim(self):
        """
        Returns the dimension of this subspace.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,10,10))
        >>> x.dim()
        4
        """

        return self._dim

    def __len__(self):
        """
        Returns the dimension of this subspace.
        Equivalent to ``self.dim()``.  This method is here because this class
        emulates an array, returning basis vectors with syntax like ``self[0]``.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,10,10))
        >>> len(x)
        4
        """

        return self._dim

    def __getitem__(self, i):
        """
        Returns a basis vector of this subspace.

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> S = TensorSubspace.from_span(np.random.randn(4,10,10))
        >>> S[0] in S
        True
        >>> S.equiv(TensorSubspace.from_span([ x for x in S ]))
        True
        """

        if self._hilb_space is None:
            return self._basis[i]
        else:
            return self._hilb_space.array(self._basis[i])

    def __contains__(self, other):
        """
        Alias for self.contains(other).

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> y = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> x in y
        False
        >>> x in y|x
        True
        >>> y[0] in x
        False
        >>> y[0] in y
        True
        """

        return self.contains(other)

    def __gt__(self, other):
        """
        Alias for self.contains(other).

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> y = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> y > x
        False
        >>> y|x > x
        True
        >>> x > y[0]
        False
        >>> y > y[0]
        True
        """

        return self.contains(other)

    def __lt__(self, other):
        """
        Alias for other.contains(self).

        >>> import numpy as np
        >>> from qitensor import TensorSubspace
        >>> x = TensorSubspace.from_span(np.random.randn(4,5,10))
        >>> y = TensorSubspace.from_span(np.random.randn(30,5,10))
        >>> x < y
        False
        >>> x < y|x
        True
        >>> #y[0] < x  # doesn't work - calls numpy's "<" operator
        """

        return other.contains(self)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
