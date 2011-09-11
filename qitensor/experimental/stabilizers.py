import numpy as np
from numpy import linalg

from qitensor import HilbertArray

epsilon = 1e-12

def get_nullspace(f, basis):
    assert len(basis) >= 0
    # f linear --> f(0)=0
    assert f(basis[0].space.array()).norm() < epsilon

    err_mat = []
    for x in basis:
        v = f(x)
        if isinstance(v, HilbertArray):
            v = v.nparray
        err_mat.append(np.concatenate([v.flatten().real, v.flatten().imag]))
    err_mat = np.array(err_mat).T
    (U, s, Vt) = linalg.svd(err_mat, full_matrices=1)
    if len(s) < len(basis):
        s.resize(len(basis))
    nullspace = Vt[abs(s[0:Vt.shape[0]]) < epsilon, :]
    assert np.isrealobj(nullspace)
    return nullspace

def basis_sl(spc):
    indices = list(spc.index_iter())
    basis = []
    for (i, l) in enumerate(indices):
        for (j, r) in enumerate(indices):
            # trace-free condition requires eliminating a basis vector
            if i==0 and j==0: continue
            for mul in (1, 1j):
                op = spc.O.array()
                op[l+r] = mul
                op -= spc.eye() * op.trace() / spc.dim()
                basis.append(op.normalized())
    return basis

def make_mostly_diagonal(mat):
    mat = linalg.qr(mat)[1]
    for (i, v) in enumerate(stab):
        idx = np.argmax(abs(v) > epsilon)
        for j in range(len(stab)):
            if j==i: continue
            stab[j] -= stab[j, idx] / v[idx] * v
        v /= linalg.norm(v)
    mat = np.array(sorted(stab, key=lambda x: np.argmax(abs(x) > epsilon)))
    return mat

def continuous_stabilizers(psi, sl_spaces, allow_phase=0, allow_scale=0):
    psi.space.assert_ket_space()

    basis = []
    slice_l = []
    slice_r = []
    for spc in sl_spaces:
        slice_l.append(len(basis))
        basis += basis_sl(spc)
        slice_r.append(len(basis))

    phase_idx = None
    if allow_phase:
        phase_idx = len(basis)
        basis.append(1j)
    scale_idx = None
    if allow_scale:
        scale_idx = len(basis)
        basis.append(1)

    #print len(basis), slice_l, slice_r

    nullspace = get_nullspace(lambda x: x*psi, basis)
    if len(nullspace) == 0:
        return []
    nullspace = make_mostly_diagonal(nullspace)
    print nullspace
    stabilizers = []
    for nvec in nullspace:
        nop = [x*y for (x, y) in zip(nvec, basis)]
        stab = []
        for (sl, sr) in zip(slice_l, slice_r):
            party_op = np.sum(nop[sl:sr])
            stab.append(party_op)
        if allow_phase:
            stab[0] += stab[0].space.eye() * nvec[phase_idx] * 1j
        if allow_scale:
            stab[0] += stab[0].space.eye() * nvec[scale_idx]
        stabilizers.append(tuple(stab))

    for stab in stabilizers:
        op = np.sum([psi.space.eye() * x for x in stab])
        assert (op * psi).norm() < epsilon
        assert (op.expm() * psi - psi).norm() < epsilon

    #return nullspace
    return stabilizers

ha=qubit('a')
hb=qudit('b',2)
hc=qudit('c',2)
hd=qudit('d',2)

ghz = (ha*hb*hc).array()
ghz[0,0,0] = 1
ghz[1,1,1] = 1
ghz.normalize()

#psi=max_entangled(ha,hb)
#stab = continuous_stabilizers(psi, [ha, hb])

#psi=(ha*hb*hc).random_array()
#stab = continuous_stabilizers(psi, [ha, hb])

stab = continuous_stabilizers((ha*hb).array([[1,0],[0,0]]), [ha, hb], 1, 1)
