#!/usr/bin/python

import qitensor
import numpy as np

from sage.all import I, exp, pi
from sage.all import matrix, block_matrix, identity_matrix
from sage.all import CommutativeRing
from qitensor import HilbertBaseField, HilbertAtom, HilbertSpace, HilbertArray

class SageHilbertBaseField(HilbertBaseField):
    def __init__(self, sage_ring):
        unique_id = 'sage '+repr(sage_ring)
        HilbertBaseField.__init__(self, object, unique_id)
        self.sage_ring = sage_ring

    def complex_unit(self):
        return self.sage_ring(I)

    def fractional_phase(self, a, b):
        return self.sage_ring(exp(2 * pi * I * a / b))

    def eye(self, size):
        return np.array(identity_matrix(self.sage_ring, size), dtype=self.dtype)

    def mat_adjoint(self, m):
        return m.sage_matrix_transform(
            lambda x: x.conjugate().transpose(), transpose_dims=True)

    def mat_inverse(self, m):
        return m.sage_matrix_transform(
            lambda x: x.inverse(), transpose_dims=True)

    def mat_det(self, m):
        return matrix(m).det()

    def mat_norm(self, m):
        return matrix(m).norm()

    def mat_conj(self, m):
        return m.sage_matrix_transform(lambda x: x.conjugate())

def can_use_type(dtype):
    return isinstance(dtype, CommutativeRing)

def create_base_field(dtype):
    assert can_use_type(dtype)
    return SageHilbertBaseField(dtype)
