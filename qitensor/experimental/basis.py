#!/usr/bin/python

# Code by Dan Stahlke (2012).

import numpy as np
import numpy.linalg as linalg

# This is the only thing that is exported.
__all__ = ['TensorBasis']

class TensorBasis(object):
    """
    >>> from tensor_basis import *
    >>> x = TensorBasis.from_span(np.random.randn(4,5,10))
    >>> x
    <TensorBasis of dim 4 over space (5, 10)>
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
    >>> y.contains(x & y)
    True
    >>> y.contains(x | y)
    False
    >>> (x | y).contains(y)
    True
    >>> y - x
    <TensorBasis of dim 26 over space (5, 10)>
    >>> (y & z).equiv(~(~y | ~z))
    True
    >>> (y-(y-x)).equiv(TensorBasis.from_span([ y.project(v) for v in x.basis ]))
    True
    """

    def __init__(self, basis, perp_basis, tol, hilb_space, validate=False):
        self.tol = tol
        self.hilb_space = hilb_space
        self.basis = basis
        self.perp_basis = perp_basis
        self.dim = basis.shape[0]
        self.perp_dim = perp_basis.shape[0]
        self.col_shp = basis.shape[1:]
        self.col_dim = np.product(self.col_shp)
        self._perp_cache = None
        # can be passed to constructor to make a space with similar configuration
        self._config_kw = { 'tol': tol, 'hilb_space': hilb_space }

        if hilb_space is not None:
            import qitensor.space
            assert isinstance(hilb_space, qitensor.space.HilbertSpace)
            assert hilb_space.shape == basis.shape[1:]

        if validate:
            assert basis.shape[1:] == perp_basis.shape[1:]
            assert self.dim + self.perp_dim == self.col_dim

        self.basis_flat = basis.reshape((self.dim, self.col_dim))
        self.perp_basis_flat = perp_basis.reshape(((self.col_dim-self.dim), self.col_dim))

        if validate:
            foo = np.tensordot(self.basis_flat.conjugate(), self.basis_flat, axes=((1,),(1,)))
            assert linalg.norm(foo - np.eye(self.dim)) < self.tol

            foo = np.tensordot(self.perp_basis_flat.conjugate(), self.perp_basis_flat, axes=((1,),(1,)))
            assert linalg.norm(foo - np.eye(self.col_dim - self.dim)) < self.tol

            foo = np.tensordot(self.basis_flat.conjugate(), self.perp_basis_flat, axes=((1,),(1,)))
            assert linalg.norm(foo) < self.tol

    @classmethod
    def from_span(cls, X, tol=1e-10, hilb_space=None, use_qr=False):
        # FIXME - accept list of HilbertArray, and auto-set hilb_space accordingly

        X = np.array(X)
        assert len(X.shape) >= 2

        col_shp = X.shape[1:]
        m = X.shape[0]
        n = np.product(col_shp)

        if hilb_space is not None:
            assert col_shp == hilb_space.shape

        if m==0:
            return cls.empty(col_shp, tol=tol, hilb_space=hilb_space)

        X = X.reshape(m, n)

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
    def empty(cls, col_shp, tol=1e-10, hilb_space=None):
        # FIXME - col_shp not needed if hilb_space given
        n = np.product(col_shp)
        basis = np.zeros((0,)+col_shp)
        perp_basis = np.eye(n).reshape((n,)+col_shp)
        return cls(basis, perp_basis, tol=tol, hilb_space=hilb_space)

    @classmethod
    def full(cls, col_shp, tol=1e-10, hilb_space=None):
        # FIXME - col_shp not needed if hilb_space given
        return cls.empty(col_shp, tol=tol, hilb_space=hilb_space).perp()

    def assert_compatible(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('TensorBasis can only add another TensorBasis')

        if self.hilb_space is not None and other.hilb_space is not None:
            assert self.hilb_space == other.hilb_space

        assert self.col_shp == other.col_shp

    def perp(self):
        """Returns orthogonal complement of this space."""
        if self._perp_cache is None:
            self._perp_cache = self.__class__(self.perp_basis, self.basis, **self._config_kw)
            self._perp_cache._perp_cache = self
        return self._perp_cache

    def __str__(self):
        if self.hilb_space is None:
            spc_str = str(self.col_shp)
        else:
            spc_str = repr(self.hilb_space)
        return "<TensorBasis of dim "+str(self.dim)+" over space ("+spc_str+")>"

    def __repr__(self):
        return str(self)

    def __invert__(self):
        """Returns orthogonal complement of this space."""
        return self.perp()

    def __or__(self, other):
        """Span of union of spaces."""
        self.assert_compatible(other)
        b_cat = np.concatenate((self.basis, other.basis), axis=0)
        return self.from_span(b_cat, **self._config_kw)

    def __and__(self, other):
        """Intersection of spaces."""
        return (self.perp() | other.perp()).perp()

    def __sub__(self, other):
        """Subspace of first space perpendicular to second space."""
        return self & other.perp()

    def to_basis(self, x):
        if self.hilb_space is not None:
            import qitensor.array
            if isinstance(x, qitensor.array.HilbertArray):
                assert x.space == self.hilb_space
                return self.to_basis(x.nparray)

        assert x.shape == self.col_shp
        nd = len(x.shape)
        return np.tensordot(self.basis.conjugate(), x, axes=(range(1, nd+1), range(nd)))

    def from_basis(self, v):
        assert len(v.shape) == 1
        assert v.shape[0] == self.dim
        ret = np.tensordot(v, self.basis, axes=((0,),(0,)))
        if self.hilb_space is None:
            return ret
        else:
            return self.hilb_space.array(ret)

    def project(self, x):
        return self.from_basis(self.to_basis(x))

    def is_perp(self, other):
        """Tests whether the given TensorBasis or vector is perpendicular to this space."""
        if isinstance(other, self.__class__):
            self.assert_compatible(other)
            foo = np.tensordot(self.basis_flat.conjugate(), other.basis_flat, axes=((1,),(1,)))
            return linalg.norm(foo) < self.tol
        else:
            return linalg.norm(self.to_basis(other)) < self.tol

    def contains(self, other):
        """Tests whether the given TensorBasis or vector is contained in this space."""
        return self.perp().is_perp(other)

    def equiv(self, other):
        return self.contains(other) and other.contains(self)

    def is_hermitian(self):
        assert len(self.col_shp) == 2
        assert self.col_shp[0] == self.col_shp[1]
        for x in self.basis:
            if not self.contains(x.conjugate().transpose()):
                return False
        return True

    def hermitian_basis(self):
        """
        Compute a basis for the Hermitian operators of this space.  Note that
        this basis is intended to map real vectors to complex operators.
        """

        assert len(self.col_shp) == 2
        assert self.col_shp[0] == self.col_shp[1]
        n = self.col_shp[0]

        # x_to_S = [ |a>, <a| ; Si ]
        x_to_S = np.zeros((n,n, self.dim), dtype=complex)
        for (S_i, Sv) in enumerate(self.basis):
            x_to_S[:, :, S_i] = Sv
        # project onto Hermitian space while simulating complex values with
        # reals on the x side
        x_to_S_1  = x_to_S    + np.transpose(x_to_S,    [1, 0, 2]).conjugate()
        x_to_S_1j = x_to_S*1j + np.transpose(x_to_S*1j, [1, 0, 2]).conjugate()
        x_to_S = np.concatenate((x_to_S_1, x_to_S_1j), axis=2)

        # decrease parameters by only taking linearly independent subspace
        sqrmat = np.array([x_to_S.real, x_to_S.imag]).reshape(2*(n**2), 2*self.dim)
        (U, s, V) = linalg.svd(sqrmat)
        n_indep = np.sum(s > self.tol)
        S_basis = U[:, :n_indep]
        x_to_S_reduced_real = S_basis.reshape(2,n,n, n_indep)
        x_to_S_reduced = x_to_S_reduced_real[0] + 1j*x_to_S_reduced_real[1]

        for i in xrange(n_indep):
            s = x_to_S_reduced[:, :, i]
            sH = s.transpose().conjugate()
            assert linalg.norm(s - sH) < 1e-13
            # correct for numerical error and make it exactly Hermitian
            x_to_S_reduced[:, :, i] = (s + sH)/2

        ret = x_to_S_reduced.transpose([2, 0, 1])

        if self.hilb_space is None:
            return ret
        else:
            # FIXME - untested
            return [self.hilb_space.array(x) for x in ret]

    def tensor_prod(self, other):
        n = len(self.basis.shape)
        m = len(other.basis.shape)

        def tp(a, b):
            foo = np.tensordot(a, b, axes=([], []))
            foo = foo.transpose([n] + range(n) + range(n+1, n+m))
            foo = foo.reshape((foo.shape[0]*foo.shape[1],) + foo.shape[2:])
            return foo

        b_b = tp(self.basis, other.basis)
        # this is simpler, but computing perp_basis manually avoids an svd/qr call
        #return self.__class__.from_span(b_b, **self._config_kw)
        bp_b  = tp(self.perp_basis, other.basis)
        b_bp  = tp(self.basis,  other.perp_basis)
        bp_bp = tp(self.perp_basis, other.perp_basis)
        b_b_p = np.concatenate((bp_b, b_bp, bp_bp), axis=0)
        return self.__class__(b_b, b_b_p, **self._config_kw)

    def map(self, f):
        b_new  = np.array([f(m) for m in self.basis])
        bp_new = np.array([f(m) for m in self.perp_basis])
        return self.__class__(b_new, bp_new, **self._config_kw)

    def transpose(self, axes):
        if self.hilb_space is not None:
            raise NotImplementedError()
        else:
            return self.map(lambda m: m.transpose(axes))

    def reshape(self, shape):
        if self.hilb_space is not None:
            raise NotImplementedError()
        else:
            return self.map(lambda m: m.reshape(shape))

    def __len__(self):
        return self.basis.shape[0]

    def __getitem__(self, i):
        if self.hilb_space is None:
            return self.basis[i]
        else:
            return self.hilb_space.array(self.basis[i])

    # FIXME - define gt, lt operators

if __name__ == "__main__":
    import doctest
    doctest.testmod()
