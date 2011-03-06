#!/usr/bin/python

import qitensor
import numpy as np
from sage.structure.sage_object import SageObject
from sage.all import *

class SageHilbertBaseField(qitensor.HilbertBaseField):
    def __init__(self, dtype, sage_ring):
        unique_id = 'sage_'+str(sage_ring)
        qitensor.HilbertBaseField.__init__(self, dtype, unique_id)
        self.sage_ring = sage_ring

    def complex_unit(self):
        return I

    def fractional_phase(self, a, b):
        return exp(2 * pi * I * a / b)

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

    def mat_norm(self, m, ord=None):
        return matrix(m).norm()

    def mat_conj(self, m):
        return m.sage_matrix_transform(lambda x: x.conjugate())

    def _atom_factory(self, label, latex_label, indices):
        return SageHilbertAtom(label, latex_label, indices, self)

    def _space_factory(self, ket_set, bra_set):
        return SageHilbertSpace(ket_set, bra_set, self)

    def _array_factory(self, space, data, noinit_data, reshape):
        return SageHilbertArray(space, data, noinit_data, reshape)

class SageHilbertAtomMixins(object):
    pass

class SageHilbertSpaceMixins(object):
    def reshaped_sage_matrix(self, m):
        return self.reshaped_np_matrix(np.array(m, dtype=self.base_field.dtype))

class SageHilbertAtom(SageHilbertAtomMixins, SageHilbertSpaceMixins, qitensor.HilbertAtom, SageObject):
    pass

class SageHilbertSpace(SageHilbertSpaceMixins, qitensor.HilbertSpace, SageObject):
    pass

class SageHilbertArray(qitensor.HilbertArray, SageObject):
    def __init__(self, space, data, noinit_data, reshape):
        qitensor.HilbertArray.__init__(self, space, data, noinit_data, reshape)

    def _matrix_(self, R=None):
        np_mat = np.array(self.as_np_matrix())

        sage_ring = self.space.base_field.sage_ring

        m = matrix(sage_ring, np_mat)
        if R is None:
            return m
        else:
            return m.change_ring(R)

    def _latex_(self):
        return '\\begin{array}{l}\n'+ \
            latex(self.space)+' \\\\\n'+ \
            latex(self.block_matrix())+ \
            '\\end{array}'

    def __repr__(self):
        return repr(self.space)+'\n'+repr(self.block_matrix())

    def __str__(self):
        return str(self.space)+'\n'+str(self.block_matrix())

    def block_matrix(self):
        hs = self.space
        
        blocks = [self]
        nrows = 1
        ncols = 1

        if len(hs.sorted_kets) > 1:
            h = hs.sorted_kets[0]
            blocks = [m[{h: i}] for m in blocks for i in h.indices]
            nrows = len(h.indices)

        if len(hs.sorted_bras) > 1:
            h = hs.sorted_bras[0]
            blocks = [m[{h: i}] for m in blocks for i in h.indices]
            ncols = len(h.indices)

        blocks = [matrix(x) for x in blocks]

        return block_matrix(blocks, nrows, ncols, True)

    def sage_matrix_transform(self, f, transpose_dims=False):
        m = matrix(self)
        m = f(m)
        out_hilb = self.space
        if transpose_dims:
            out_hilb = out_hilb.H
        return out_hilb.reshaped_sage_matrix(m)

sr_base_field = SageHilbertBaseField(object, SR)
cc_base_field = SageHilbertBaseField(object, CC)
cdf_base_field = SageHilbertBaseField(object, CDF)

# convenience functions

def lookup_base_field(base_field):
    if isinstance(base_field, qitensor.HilbertBaseField):
        return base_field
    elif base_field == SR:
        return sr_base_field
    elif base_field == CC:
        return cc_base_field
    elif base_field == CDF:
        return cdf_base_field

def indexed_space(label, indices, base_field=SR, latex_label=None):
    return lookup_base_field(base_field).indexed_space(label, indices, latex_label)

def qudit(label, dim, base_field=SR, latex_label=None):
    return lookup_base_field(base_field).qudit(label, dim, latex_label)

def qubit(label, base_field=SR, latex_label=None):
    return lookup_base_field(base_field).qubit(label, latex_label)
