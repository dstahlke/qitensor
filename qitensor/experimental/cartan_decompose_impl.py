# The qitensor frontend for this module is in cartan_decompose.py

# If I remember correctly, this was based on
# PHYSICAL REVIEW A, VOLUME 63, 062309
# Optimal creation of entanglement using a two-qubit gate
# B. Kraus and J. I. Cirac
# http://pra.aps.org/abstract/PRA/v63/i6/e062309

# A simpler way of finding alpha (but not {UV}{AB}?) is given in
# http://pra.aps.org/abstract/PRA/v68/i5/e052311
# which also discusses how to take the right branch of the logarithm
# so that pi/4 >= alpha_x >= alpha_y >= |alpha_z|.
# FIXME - this should be implemented

# A much more complicated treatment, applicable to multiple qubits, is given in
# http://arxiv.org/abs/quant-ph/0010100

from __future__ import print_function

import numpy as np
# FIXME - why doesn't scipy work here?
from numpy import linalg

def bipartite_op_tensor_product(A, B):
    """Tensor product of two 2x2 matrices into a 4x4 matrix."""
    return np.matrix(np.tensordot(A, B, axes=([],[])).
        transpose((0,2,1,3)).reshape(4, 4))

def bipartite_state_tensor_product(A, B):
    """Tensor product of two 2-dim vectors into a 4-dim vector."""
    return np.matrix(np.tensordot(A, B, axes=([],[])).reshape(4,1))

MAGIC_BASIS = np.matrix([
    [1, 0, 0, 1],
    [-1j, 0, 0, 1j],
    [0, 1, -1, 0],
    [0, -1j, -1j, 0]
], dtype=complex).conj() / np.sqrt(2)

PAULIS = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]],
])

def _make_double_paulis():
    return np.array([
        bipartite_op_tensor_product(PAULIS[i], PAULIS[i]) for i in range(3)
    ])
DOUBLE_PAULIS = _make_double_paulis()

def decompose_dyad(dyad):
    """Decompose a dyad into a ket and a bra."""

    (U, s, VT) = linalg.svd(dyad)
    # make sure it is rank 1
    assert np.allclose(s, [1, 0])
    ket = np.matrix(U [:, 0])
    bra = np.matrix(VT[0, :])
    return (ket, bra)

def branch_cut(lam):
    # Attempt at implementing branch cut choice described in
    # http://pra.aps.org/abstract/PRA/v68/i5/e052311

    tol = 1e-10
    # Wrap lam into the (-pi/4, 3pi/4] range.
    for i in range(4):
        lam[i] %= np.pi
        if lam[i] > 3*np.pi/4:
            lam[i] -= np.pi
    nn = np.sum(lam) / np.pi
    assert abs(nn - int(round(nn))) < tol
    for i in range(int(round(nn))):
        lam[3-i] -= np.pi

def locally_transform_to_magic_basis(psi):
    """Find local unitaries and phases that transform a maximally
    entangled basis psi to the magic basis"""

    tol = 1e-10

    # make sure input is a complex numpy matrix
    psi = np.matrix(psi, dtype=complex)

    # check to make sure input states really are fully entangled
    for i in range(4):
        (_u, schmidt, _v) = linalg.svd(psi[:, i].reshape(2, 2))
        assert np.allclose(schmidt, [1/np.sqrt(2), 1/np.sqrt(2)])

    # make coeffs real in magic basis
    phases = np.zeros(4, dtype=complex)
    for i in range(4):
        psi_in_magic = MAGIC_BASIS * psi
        j = np.argmax(abs(psi_in_magic[:, i]))
        v = psi_in_magic[j, i]
        phases[i] = v / abs(v)
    total_phase = np.product(phases) ** (1.0/4.0)
    phases /= total_phase
    gamma = -np.angle(phases)
    branch_cut(gamma)
    psi_bar = psi * np.diag(np.exp(1j * gamma))

    separable = []
    # By my reckoning, it seems that the plusses and minuses are flipped in the
    # paper.
    separable.append((psi_bar[:,0] + 1j*psi_bar[:,1])/np.sqrt(2))
    separable.append((psi_bar[:,0] - 1j*psi_bar[:,1])/np.sqrt(2))
    separable.append((psi_bar[:,2] + 1j*psi_bar[:,3])/np.sqrt(2))
    separable.append((psi_bar[:,2] - 1j*psi_bar[:,3])/np.sqrt(2))

    # check to make sure these really are separable
    for i in range(4):
        (_u, schmidt, _v) = linalg.svd(separable[i].reshape(2, 2))
        assert np.allclose(schmidt, [1, 0])

    (e, f)           = decompose_dyad(separable[0].reshape(2, 2))
    (e_perp, f_perp) = decompose_dyad(separable[1].reshape(2, 2))
    f = f.T
    f_perp = f_perp.T
    assert e.H * e_perp < tol
    assert f.H * f_perp < tol

    e_f = bipartite_state_tensor_product(e, f)
    eperp_f = bipartite_state_tensor_product(e_perp, f)
    e_fperp = bipartite_state_tensor_product(e, f_perp)
    eperp_fperp = bipartite_state_tensor_product(e_perp, f_perp)

    delta = np.angle(e_fperp.H * psi_bar[:,3] * 1j * np.sqrt(2))
    # convert to scalar
    delta = delta[0,0]

    argmax3 = np.argmax(abs(psi_bar[:, 2]))
    sign3 = psi_bar[argmax3, 2] / (1/np.sqrt(2)*
        (np.exp(1j*delta)*e_fperp[argmax3, 0] -
        np.exp(-1j*delta)*eperp_f[argmax3, 0]))
    #print "s3", sign3
    if abs(sign3-1) < tol:
        pass
    elif abs(sign3+1) < tol:
        gamma[2] += np.pi
    else:
        assert 0

    assert np.allclose(psi_bar[:,0], 1/np.sqrt(2)*(e_f + eperp_fperp))
    assert np.allclose(psi_bar[:,1], (-1j)/np.sqrt(2)*(e_f - eperp_fperp))
    # These appear to be switched in the paper...
    assert np.allclose(psi_bar[:,2],
        sign3/np.sqrt(2)*(np.exp(1j*delta)*e_fperp - np.exp(-1j*delta)*eperp_f))
    assert np.allclose(psi_bar[:,3],
        (-1j)/np.sqrt(2)*(np.exp(1j*delta)*e_fperp + np.exp(-1j*delta)*eperp_f))

    UA = np.matrix(np.zeros((2, 2), dtype=complex))
    UA[0, :] = e.H
    UA[1, :] = e_perp.H * np.exp(1j * delta)
    UB = np.matrix(np.zeros((2, 2), dtype=complex))
    UB[0, :] = f.H
    UB[1, :] = f_perp.H * np.exp(-1j * delta)

    return (UA, UB, gamma)

def unitary_to_cartan(U):
    """
    Decomposes a bipartite qubit unitary (given as a 4x4 matrix) into the form
    $(U_A \otimes U_B) e^{-i \alpha_i \sigma_{Ai} \otimes \sigma_{Bi}} (V_A \otimes V_B)$
    where each alpha_i is in the (-pi/4, pi/4] range.
    Five values are returned from this function: (UA, UB, VA, VB, alpha).
    """

    #print "---"
    # make sure input is a complex numpy matrix
    U = np.matrix(U, dtype=complex)
    assert U.shape == (4, 4)

    tol = 1e-10

    total_phase_sq = linalg.det(U) ** (1.0/2.0)
    assert abs(abs(total_phase_sq) - 1) < tol

    mb = MAGIC_BASIS
    UT = mb.H * (mb * U * mb.H).T * mb
    UTU = UT * U

    # FIXME: running eig in magic basis hopefully ensures that degenerate
    # eigenvectors are fully entangled basis states
    UTU = mb * UTU * mb.H
    (ew, psi) = linalg.eig(UTU)
    # Make eigenvectors orthonormal.  This is needed in cases where there are
    # degenerate eigenvalues.
    # FIXME: out of luck, this seems to work, when done in the magic basis.
    # Really what needs to be done is to find the basis for degenerate spaces
    # in which each eigenvector is real up to total phase.
    #print "before:", psi
    for i in [1,2,3]:
        for j in range(i):
            dp = (psi[:, j].H * psi[:, i])[0, 0]
            psi[:, i] -= dp * psi[:, j]
        psi[:, i] /= linalg.norm(psi[:, i])

    # Make sure eigenvectors are real.
    for i in [1,2,3]:
        j = np.argmax(abs(psi[:, i]))
        assert abs(psi[j, i]) > 0
        phase = psi[j, i] / abs(psi[j, i])
        psi[:, i] /= phase
    assert np.allclose(psi.imag, 0)

    #print "after:", psi
    # Change back to computational basis
    psi = mb.H * psi

    ew /= total_phase_sq
    assert np.allclose(np.abs(ew), 1)
    epsilon = np.angle(ew) / 2.0

    branch_cut(epsilon)

    (VA, VB, xi) = locally_transform_to_magic_basis(psi)

    psi_tilde = U * psi * np.matrix(np.diag(np.exp(-1j*epsilon)))
    (UA, UB, zeta) = locally_transform_to_magic_basis(psi_tilde)

    lam = zeta - xi - epsilon
    assert np.allclose(lam.imag, 0)
    lam = lam.real

    #print "epsilon", epsilon/np.pi, "sum", np.sum(epsilon)/np.pi
    #print "xi", xi/np.pi, "sum", np.sum(xi)/np.pi
    #print "zeta", zeta/np.pi, "sum", np.sum(zeta)/np.pi
    #print "lam", lam/np.pi, "sum", np.sum(lam)/np.pi

    UA *= np.exp(1j*sum(lam)/4)
    lam -= sum(lam)/4
    UA = UA.H
    UB = UB.H

    alpha = np.array([
        (lam[0]+lam[3])/2,
        (lam[1]+lam[3])/2,
        (lam[0]+lam[1])/2,
    ])

    #print "alpha", alpha / np.pi

    # transform alpha to (-pi/4, pi/4] range
    for i in range(3):
        alpha[i] %= 2*np.pi
        while alpha[i] > np.pi/4:
            alpha[i] -= np.pi/2
            VA = np.matrix(PAULIS[i]) * VA * (-1j)
            VB = np.matrix(PAULIS[i]) * VB

    do_verify = True
    if do_verify:
        # make sure alphas are in the right range
        for i in range(3):
            assert -np.pi/4 <= alpha[i] <= np.pi/4
        # make sure the local unitaries really are unitary
        for op in (UA, UB, VA, VB):
            assert np.allclose(op.H * op, np.eye(2, 2))
            assert np.allclose(op * op.H, np.eye(2, 2))
        # make sure that we get back the original bipartite unitary
        Ud = unitary_from_cartan(alpha)
        UAB = bipartite_op_tensor_product(UA, UB)
        VAB = bipartite_op_tensor_product(VA, VB)
        assert np.allclose(UAB * Ud * VAB, U)

    return (UA, UB, VA, VB, alpha)

def unitary_from_cartan(alpha):
    """
    Returns $e^{-i \alpha_i \sigma_{Ai} \otimes \sigma_{Bi}}$.
    """
    assert len(alpha) == 3

    (Ux, Uy, Uz) = [np.matrix(
        np.cos(alpha[i])*np.eye(4, 4) - 1j * np.sin(alpha[i])*DOUBLE_PAULIS[i]
        ) for i in range(3) ]

    return Ux * Uy * Uz

if __name__ == "__main__":
    # Round off numbers that are very close to zero when printing the results.
    np.set_printoptions(suppress=True)

    M = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0],
    ])

    (UA, UB, VA, VB, alpha) = unitary_to_cartan(M)

    print('=== UA ===\n', UA)
    print('=== UB ===\n', UB)
    print('=== VA ===\n', VA)
    print('=== VB ===\n', VB)
    print('=== alpha/pi ===\n', alpha / np.pi)

    # Test the answer to make sure it is correct.
    # First, compute exp(-i alpha_k \sigma_k \ot \sigma_k)
    Q = unitary_from_cartan(alpha)
    # Turn the 4x4 matrix into a 2x2x2x2 tensor.
    Q = Q.A.reshape((2,2,2,2))
    # Compute (UA \ot UB) Q (VA \ot VB)
    R = np.einsum('ab,cd,bdwy,wx,yz->acxz', UA, UB, Q, VA, VB)
    # Convert 2x2x2x2 tensor into 4x4
    R = R.reshape((4,4))
    # Make sure this equals the original unitary
    assert linalg.norm(R - M) < 1e-12
