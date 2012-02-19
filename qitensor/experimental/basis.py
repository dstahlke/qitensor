#!/usr/bin/python

import numpy as np
import numpy.linalg as linalg

# This is the only thing that is exported.
__all__ = ['TensorBasis']

class TensorBasis(object):
    """
    Represents a basis of a tensor space.  Methods are available for projecting
    to this basis, computing span and intersection of bases, etc.  This module
    can be used independently of qitensor, and the doctest below reflects this.
    However, if you are using qitensor then typically a basis would be created
    using one of the following methods:

    * :func:`qitensor.array.HilbertArray.span`
    * :func:`qitensor.space.HilbertSpace.full_space`
    * :func:`qitensor.space.HilbertSpace.empty_space`

    >>> import numpy as np
    >>> from qitensor.experimental.basis import TensorBasis
    >>> x = TensorBasis.from_span(np.random.randn(4,5,10))
    >>> x
    <TensorBasis of dim 4 over space (5, 10)>
    >>> x.dim()
    4
    >>> x[0] in x
    True
    >>> x.perp()[0] in x
    False
    >>> y = TensorBasis.from_span(np.random.randn(30,5,10))
    >>> y
    <TensorBasis of dim 30 over space (5, 10)>
    >>> z = TensorBasis.from_span(np.random.randn(30,5,10))
    >>> z
    <TensorBasis of dim 30 over space (5, 10)>
    >>> x | y
    <TensorBasis of dim 34 over space (5, 10)>
    >>> y | z
    <TensorBasis of dim 50 over space (5, 10)>
    >>> y & z
    <TensorBasis of dim 10 over space (5, 10)>
    >>> y > x&y
    True
    >>> y > x|y
    False
    >>> x|y > y
    True
    >>> y - x
    <TensorBasis of dim 26 over space (5, 10)>
    >>> (y & z).equiv(~(~y | ~z))
    True
    >>> (y-(y-x)).equiv(TensorBasis.from_span([ y.project(v) for v in x ]))
    True
    """

    def __init__(self, basis, perp_basis, tol, hilb_space, validate=False):
        self._tol = tol
        self._hilb_space = hilb_space
        self._basis = basis
        self._perp_basis = perp_basis
        self._dim = basis.shape[0]
        self._col_shp = basis.shape[1:]
        self._col_dim = np.product(self._col_shp)
        self._perp_cache = None
        self._hermit_cache = None
        # can be passed to constructor to make a space with similar configuration
        self._config_kw = { 'tol': tol, 'hilb_space': hilb_space }

        if hilb_space is not None:
            import qitensor.space
            assert isinstance(hilb_space, qitensor.space.HilbertSpace)
            assert hilb_space.shape == basis.shape[1:]

        if validate:
            assert basis.shape[1:] == perp_basis.shape[1:]
            assert self._dim + perp_basis.shape[0] == self._col_dim

        self._basis_flat = basis.reshape((self._dim, self._col_dim))
        self._perp_basis_flat = perp_basis.reshape(((self._col_dim-self._dim), self._col_dim))

        if validate:
            foo = np.tensordot(self._basis_flat.conjugate(), self._basis_flat, axes=((1,),(1,)))
            assert linalg.norm(foo - np.eye(self._dim)) < self._tol

            foo = np.tensordot(self._perp_basis_flat.conjugate(), self._perp_basis_flat, axes=((1,),(1,)))
            assert linalg.norm(foo - np.eye(self._col_dim - self._dim)) < self._tol

            foo = np.tensordot(self._basis_flat.conjugate(), self._perp_basis_flat, axes=((1,),(1,)))
            assert linalg.norm(foo) < self._tol

    @classmethod
    def from_span(cls, X, tol=1e-10, hilb_space=None, dtype=None):
        if not isinstance(X, np.ndarray):
            X = list(X)
            try:
                import qitensor.array
                if isinstance(X[0], qitensor.array.HilbertArray):
                    assert hilb_space is None or hilb_space == X[0].space
                    hilb_space = X[0].space
                    for op in X:
                        assert isinstance(op, qitensor.array.HilbertArray)
                        assert op.space == hilb_space
                    X = [op.nparray for op in X]
            except ImportError:
                pass

        X = np.array(X, dtype=dtype)
        assert len(X.shape) >= 2

        col_shp = X.shape[1:]
        m = X.shape[0]
        n = np.product(col_shp)

        if hilb_space is not None:
            assert col_shp == hilb_space.shape

        if m==0:
            return cls.empty(col_shp, tol=tol, hilb_space=hilb_space)

        X = X.reshape(m, n)

        use_qr = False
        if use_qr:
            # I think that sometimes QR doesn't do exactly what I want.  The code
            # below potentially includes extra vectors into the basis.
            assert 0
            # Append random values to that Y represents the whole space.
            # This allows us to also generate the basis orthogonal to span{X}.
            Y = np.concatenate((X, np.random.rand(n, n)), axis=0)
            (Q, R) = linalg.qr(Y.transpose(), mode='full')
            dim = np.sum([linalg.norm(row) > tol for row in R[:, :m]])
            basis      = Q[:, :dim].transpose().reshape((dim,)+col_shp)
            perp_basis = Q[:, dim:].transpose().reshape((n-dim,)+col_shp)
        else:
            (_U, s, V) = linalg.svd(X, full_matrices=True)
            dim = np.sum(s > tol)
            basis      = V[:dim, :].reshape((dim,)+col_shp)
            perp_basis = V[dim:, :].reshape((n-dim,)+col_shp)

        return cls(basis, perp_basis, tol=tol, hilb_space=hilb_space)

    @classmethod
    def empty(cls, col_shp, tol=1e-10, hilb_space=None, dtype=complex):
        # FIXME - col_shp not needed if hilb_space given
        n = np.product(col_shp)
        basis = np.zeros((0,)+col_shp, dtype=dtype)
        perp_basis = np.eye(n).reshape((n,)+col_shp)
        return cls(basis, perp_basis, tol=tol, hilb_space=hilb_space)

    @classmethod
    def full(cls, col_shp, tol=1e-10, hilb_space=None, dtype=complex):
        # FIXME - col_shp not needed if hilb_space given
        return cls.empty(col_shp, tol=tol, hilb_space=hilb_space, dtype=dtype).perp()

    def assert_compatible(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('other object is not a TensorBasis')

        if self._hilb_space is not None and other._hilb_space is not None:
            assert self._hilb_space == other._hilb_space

        assert self._col_shp == other._col_shp

    def perp(self):
        """Returns orthogonal complement of this space."""
        if self._perp_cache is None:
            self._perp_cache = self.__class__(self._perp_basis, self._basis, **self._config_kw)
            self._perp_cache._perp_cache = self
        return self._perp_cache

    def __str__(self):
        if self._hilb_space is None:
            spc_str = str(self._col_shp)
        else:
            spc_str = '('+repr(self._hilb_space)+')'
        return "<TensorBasis of dim "+str(self._dim)+" over space "+spc_str+">"

    def __repr__(self):
        return str(self)

    def __invert__(self):
        """Returns orthogonal complement of this space."""
        return self.perp()

    def __or__(self, other):
        """Span of union of spaces."""
        self.assert_compatible(other)
        b_cat = np.concatenate((self._basis, other._basis), axis=0)
        return self.from_span(b_cat, **self._config_kw)

    def __and__(self, other):
        """Intersection of spaces."""
        return (self.perp() | other.perp()).perp()

    def __sub__(self, other):
        """Subspace of first space perpendicular to second space."""
        return self & other.perp()

    def to_basis(self, x):
        if self._hilb_space is not None:
            import qitensor.array
            if isinstance(x, qitensor.array.HilbertArray):
                assert x.space == self._hilb_space
                return self.to_basis(x.nparray)

        assert x.shape == self._col_shp
        nd = len(x.shape)
        return np.tensordot(self._basis.conjugate(), x, axes=(range(1, nd+1), range(nd)))

    def from_basis(self, v):
        assert len(v.shape) == 1
        assert v.shape[0] == self._dim
        ret = np.tensordot(v, self._basis, axes=((0,),(0,)))
        if self._hilb_space is None:
            return ret
        else:
            return self._hilb_space.array(ret)

    def project(self, x):
        return self.from_basis(self.to_basis(x))

    def is_perp(self, other):
        """Tests whether the given TensorBasis or vector is perpendicular to this space."""
        if isinstance(other, self.__class__):
            self.assert_compatible(other)
            foo = np.tensordot(self._basis_flat.conjugate(), other._basis_flat, axes=((1,),(1,)))
            return linalg.norm(foo) < self._tol
        else:
            return linalg.norm(self.to_basis(other)) < self._tol

    def contains(self, other):
        """Tests whether the given TensorBasis or vector is contained in this space."""
        return self.perp().is_perp(other)

    def equiv(self, other):
        return self.contains(other) and other.contains(self)

    def is_hermitian(self):
        assert len(self._col_shp) == 2
        assert self._col_shp[0] == self._col_shp[1]
        for x in self._basis:
            if not self.contains(x.conjugate().transpose()):
                return False
        return True

    def hermitian_basis(self):
        """
        Compute a basis for the Hermitian operators of this space.  Note that
        this basis is intended to map real vectors to complex operators.
        """

        if self._hermit_cache is not None:
            return self._hermit_cache

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
        (U, s, V) = linalg.svd(sqrmat)
        n_indep = np.sum(s > self._tol)
        S_basis = U[:, :n_indep]
        x_to_S_reduced_real = S_basis.reshape(2,n,n, n_indep)
        x_to_S_reduced = x_to_S_reduced_real[0] + 1j*x_to_S_reduced_real[1]

        for i in xrange(n_indep):
            s = x_to_S_reduced[:, :, i]
            sH = s.transpose().conjugate()
            assert linalg.norm(s - sH) < 1e-13
            # correct for numerical error and make it exactly Hermitian
            x_to_S_reduced[:, :, i] = (s + sH)/2

        hbasis = x_to_S_reduced.transpose([2, 0, 1])

        if self._hilb_space is None:
            self._hermit_cache = hbasis
        else:
            # FIXME - untested
            self._hermit_cache = [self._hilb_space.array(x) for x in hbasis]

        return self._hermit_cache

    def tensor_prod(self, other):
        # FIXME - implement this.  It involves reshuffling the indices and
        # making sure the spaces aren't duplicated.
        if self._hilb_space is not None:
            raise NotImplementedError()

        n = len(self._basis.shape)
        m = len(other._basis.shape)

        def tp(a, b):
            foo = np.tensordot(a, b, axes=([], []))
            foo = foo.transpose([n] + range(n) + range(n+1, n+m))
            foo = foo.reshape((foo.shape[0]*foo.shape[1],) + foo.shape[2:])
            return foo

        b_b = tp(self._basis, other._basis)
        # this is simpler, but computing perp_basis manually avoids an svd/qr call
        #return self.__class__.from_span(b_b, **self._config_kw)
        bp_b  = tp(self._perp_basis, other._basis)
        b_bp  = tp(self._basis,  other._perp_basis)
        bp_bp = tp(self._perp_basis, other._perp_basis)
        b_b_p = np.concatenate((bp_b, b_bp, bp_bp), axis=0)
        return self.__class__(b_b, b_b_p, **self._config_kw)

    def map(self, f):
        b_new = [f(m) for m in self]
        if self._hilb_space is not None:
            cfg = self._config_kw.copy()
            cfg['hilb_space'] = b_new[0].space
        else:
            cfg = self._config_kw
        return self.__class__.from_span(b_new, **cfg)

    def _nomath_map(self, f):
        """
        Like map, but assumes the operation preserves orthogonality.
        """
        b_new  = np.array([f(m) for m in self._basis])
        bp_new = np.array([f(m) for m in self._perp_basis])
        return self.__class__(b_new, bp_new, **self._config_kw)

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
        Returns a random vector in this basis.

        >>> import numpy as np
        >>> from qitensor.experimental.basis import TensorBasis
        >>> x = TensorBasis.from_span(np.random.randn(4,5,10))
        >>> x
        <TensorBasis of dim 4 over space (5, 10)>
        >>> v = x.random_vec()
        >>> v in x
        True
        >>> abs(1 - np.linalg.norm(v)) < 1e-13
        True
        >>> x.equiv(TensorBasis.from_span([ x.random_vec() for i in range(x.dim()) ]))
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

    def dim(self):
        return self._dim

    def __len__(self):
        return self._dim

    def __getitem__(self, i):
        if self._hilb_space is None:
            return self._basis[i]
        else:
            return self._hilb_space.array(self._basis[i])

    def __contains__(self, other):
        """
        Alias for self.contains(other).
        """
        return self.contains(other)

    def __gt__(self, other):
        """
        Alias for self.contains(other).
        """
        return self.contains(other)

    def __lt__(self, other):
        """
        Alias for other.contains(self).
        """
        return other.contains(self)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
