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

    def eye(self, size):
        return np.array(sage.all.identity_matrix(self.sage_ring, size), dtype=self.dtype)

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
        return self.sqrt(np.sum(m.nparray ** 2))

    def mat_conj(self, m):
        return m.sage_matrix_transform(lambda x: x.conjugate())

def can_use_type(dtype):
    return isinstance(dtype, sage.all.CommutativeRing)

def create_base_field(dtype):
    assert can_use_type(dtype)
    return SageHilbertBaseField(dtype)
