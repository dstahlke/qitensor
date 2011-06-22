#!/usr/bin/python

import qitensor
import numpy as np

import sage.all
from qitensor import HilbertBaseField, HilbertAtom, HilbertSpace, HilbertArray

class SageHilbertBaseField(HilbertBaseField):
    def __init__(self, sage_ring):
        unique_id = 'sage '+repr(sage_ring)
        HilbertBaseField.__init__(self, object, unique_id)
        self.sage_ring = sage_ring

    def complex_unit(self):
        return self.sage_ring(sage.all.I)

    def input_cast_function(self):
        return lambda x: self.sage_ring(x)

    def fractional_phase(self, a, b):
        return self.sage_ring(sage.all.exp(2 * sage.all.pi * sage.all.I * a / b))

    def sqrt(self, x):
        return self.sage_ring(sage.all.sqrt(x))

    def xlog2x(self, x):
        return self.sage_ring(0 if x<=0 else x*sage.all.log(x)/sage.all.log2)

    def eye(self, size):
        return np.array(sage.all.identity_matrix(self.sage_ring, size), dtype=self.dtype)

    def mat_n(self, m, prec=None, digits=None):
        return m.sage_matrix_transform(
            lambda x: x.n(prec=prec, digits=digits))

    def mat_simplify(self, m, full=False):
        if full:
            return m.apply_map(lambda x: x.simplify_full())
        else:
            return m.apply_map(lambda x: x.simplify())

    def mat_adjoint(self, m):
        return m.sage_matrix_transform(
            lambda x: x.conjugate().transpose(), transpose_dims=True)

    def mat_inverse(self, m):
        return m.sage_matrix_transform(
            lambda x: x.inverse(), transpose_dims=True)

    def mat_det(self, m):
        return sage.all.matrix(m).det()

    def mat_norm(self, m):
        # Sage's matrix norm doesn't work for SR (it casts to CDF)
        #return sage.all.matrix(m).norm()
        return self.sqrt(np.sum(m.nparray * np.conj(m.nparray)))

    def mat_conj(self, m):
        return m.sage_matrix_transform(lambda x: x.conjugate())

    def mat_pow(self, m, n):
        return m.sage_matrix_transform(lambda x: x**n)

    def mat_eig(self, m, w_space, hermit):
        w_space.assert_ket_space()
        (w, v) = sage.all.matrix(m).eigenmatrix_right()

        # convert to numpy to allow sorting
        w = np.array(w.diagonal())
        v = np.array(v)

        # sort eigenvalues in ascending order of real component
        srt = np.argsort(-w)
        w = w[srt]
        v = v[:, srt]

        # Sage doesn't normalize the columns for symbolic expressions, so do it
        # here.
        v = np.array([c / sage.all.sqrt(np.sum(c**2)) for c in v.T]).T

        W = (w_space * w_space.H).diag(w)
        V = (m.space.ket_space() * w_space.H).reshaped_np_matrix(v)
        return (W, V)

    def mat_eigvals(self, m, hermit):
        w = sage.all.matrix(m).eigenvalues()

        # sort eigenvalues in ascending order of real component
        w = -np.sort(-np.array(w))

        if hermit:
            assert np.all(np.imag(w) == 0)
            w = np.real(w)

        return w

def can_use_type(dtype):
    return isinstance(dtype, sage.all.CommutativeRing)

def create_base_field(dtype):
    assert can_use_type(dtype)
    return SageHilbertBaseField(dtype)
