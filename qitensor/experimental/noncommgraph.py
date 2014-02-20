# Noncommutative graphs as defined by Duan, Severini, Winter in arXiv:1002.2514.

import numpy as np
import scipy.linalg as linalg
import itertools
import cvxopt.base
import cvxopt.solvers

from qitensor import qudit, HilbertSpace
from qitensor.superop import CP_Map
from qitensor.subspace import TensorSubspace

# This is the only thing that is exported.
__all__ = ['NoncommutativeGraph']

### Some helper functions for cvxopt ###

def mat_cplx_to_real(cmat):
    #rmat = np.zeros((2, cmat.shape[0], 2, cmat.shape[1]))
    #rmat[0, :, 0, :] = cmat.real
    #rmat[1, :, 1, :] = cmat.real
    #rmat[0, :, 1, :] = -cmat.imag
    #rmat[1, :, 0, :] = cmat.imag
    ## preserve the norm
    #rmat /= np.sqrt(2)
    #return rmat.reshape(cmat.shape[0]*2, cmat.shape[1]*2)
    return np.bmat([[cmat.real, -cmat.imag], [cmat.imag, cmat.real]]) / np.sqrt(2)

def mat_real_to_cplx(rmat):
    w = rmat.shape[0] // 2
    h = rmat.shape[1] // 2
    return (rmat[:w,:h] + rmat[w:,h:] + 1j*rmat[w:,:h] - 1j*rmat[:w,h:]) / np.sqrt(2)

def make_F_real(Fx_list, F0_list):
    '''
    Convert F0, Fx arrays to real if needed, by considering C as a vector space
    over R.  This is needed because cvxopt cannot handle complex inputs.
    '''

    F0_list_real = []
    Fx_list_real = []
    for (F0, Fx) in zip(F0_list, Fx_list):
        if F0.dtype.kind == 'c' or Fx.dtype.kind == 'c':
            F0_list_real.append(mat_cplx_to_real(F0))

            mr = np.zeros((Fx.shape[0]*2, Fx.shape[1]*2, Fx.shape[2]))
            for i in range(Fx.shape[2]):
                mr[:, :, i] = mat_cplx_to_real(Fx[:, :, i])
            Fx_list_real.append(mr)
        else:
            F0_list_real.append(F0)
            Fx_list_real.append(Fx)

    assert len(F0_list_real) == len(F0_list)
    assert len(Fx_list_real) == len(Fx_list)
    return (Fx_list_real, F0_list_real)

def call_sdp(c, Fx_list, F0_list):
    '''
    Solve the SDP which minimizes $c^T x$ under the constraint
    $\sum_i Fx_i x_i - F0 \ge 0$ for all (Fx, F0) in (Fx_list, F0_list).
    '''

    # Alternatively, the SDPA library can be used, but this requires
    # interfacing to C libraries.
    #xvec = sdpa.run_sdpa(c, Fx_list, F0_list).

    for (k, (F0, Fx)) in enumerate(zip(F0_list, Fx_list)):
        assert linalg.norm(F0 - F0.conj().T) < 1e-10
        for i in range(Fx.shape[2]):
            assert linalg.norm(Fx[:,:,i] - Fx[:,:,i].conj().T) < 1e-10

    # Note: Fx and F0 must be negated when passed to cvxopt.sdp.
    # FIXME - move this negation outwards, to avoid confusion
    (Fx_list, F0_list) = make_F_real(Fx_list, F0_list)
    Gs = [cvxopt.base.matrix(-Fx.reshape(Fx.shape[0]**2, Fx.shape[2])) for Fx in Fx_list]
    hs = [cvxopt.base.matrix(-F0) for F0 in F0_list]

    sol = cvxopt.solvers.sdp(cvxopt.base.matrix(c), Gs=Gs, hs=hs)
    xvec = np.array(sol['x']).flatten()

    sol['Gs'] = Gs
    sol['hs'] = hs

    if sol['status'] == 'optimal':
        for (G, h) in zip(Gs, hs):
            G = np.array(G)
            h = np.array(h)
            M = np.dot(G, xvec).reshape(h.shape)
            assert linalg.eigvalsh(h-M)[0] > -1e-7

    return (xvec, sol)

def check_psd(M):
    """
    By how much does M fail to be PSD?
    """

    if isinstance(M, HilbertArray):
        M = M.nparray

    if len(M.shape) == 4:
        M = M.reshape(M.shape[0]*M.shape[1], M.shape[2]*M.shape[3])

    err_H = linalg.norm(M - M.T.conj())
    err_P = linalg.eigvalsh(M + M.T.conj())[0]
    err_P = 0 if err_P > 0 else -err_P

    return err_H + err_P

def project_dh(M):
    """
    Get doubly Hermitian component.
    """
    M = (M + M.transpose(1,0,3,2).conj()) / 2
    M = (M + M.transpose(2,3,0,1).conj()) / 2
    return M

### The main code ######################

class NoncommutativeGraph(object):
    """Non-commutative graphs as described in arXiv:1002.2514."""

    def __init__(self, S):
        """
        Create a non-commutative graph from provided TensorSubspace.
        """

        assert S.is_hermitian()

        self.S = S

        # Make it a space over rank-2 tensors.
        S_flat = S._op_flatten()
        assert S_flat._col_shp[0] == S_flat._col_shp[1]
        n = self.n = S_flat._col_shp[0]

        self.S_basis  = np.array(S_flat.hermitian_basis()) \
                if S_flat.dim() else np.zeros((0, n, n), dtype=complex)
        self.Sp_basis = np.array(S_flat.perp().hermitian_basis()) \
                if S_flat.perp().dim() else np.zeros((0, n, n), dtype=complex)
        assert len(self.S_basis.shape) == 3
        assert len(self.Sp_basis.shape) == 3

        assert np.eye(n) in S_flat

        self.Y_basis = self._get_S_ot_L_basis(self.S_basis)
        self.Y_basis_dh = self._basis_doubly_hermit(self.S_basis)
        self.T_basis = self._get_S_ot_L_basis(self.Sp_basis)
        self.T_basis_dh = self._basis_doubly_hermit(self.Sp_basis)
        self.full_basis_dh = self._basis_doubly_hermit(TensorSubspace.full((n,n)).hermitian_basis())

        self.cond_psd = {
            'name': 'psd',
            'basis': self.full_basis_dh,
            'R':  lambda Z: Z.transpose((0,2,1,3)).reshape(n**2, n**2),
            'R*': lambda Z: Z.reshape(n,n,n,n).transpose((0,2,1,3)),
            '0': np.zeros((n**2, n**2), dtype=complex),
        }

        self.cond_ppt = {
            'name': 'ppt',
            'basis': self.full_basis_dh,
            'R':  lambda Z: Z.transpose((1,2,0,3)).reshape(n**2, n**2),
            'R*': lambda Z: Z.reshape(n,n,n,n).transpose((2,0,1,3)),
            '0': np.zeros((n**2, n**2), dtype=complex),
        }

        for c in [self.cond_psd, self.cond_ppt]:
            m = np.random.random((n**2, n**2))
            m2 = c['R'](c['R*'](m))
            assert linalg.norm(m-m2) < 1e-10

    def __str__(self):
        return '<NoncommutativeGraph of '+self.S._str_inner()+'>'

    def __repr__(self):
        return str(self)

    @classmethod
    def from_adjmat(cls, adj_mat):
        """
        Create a non-commutative graph from the adjacency matrix of a classical graph.

        The given adjacency matrix must be symmetric.

        >>> from noncommgraph import NoncommutativeGraph
        >>> import numpy
        >>> # 5-cycle graph
        >>> adj_mat = np.array([
        ...     [1, 1, 0, 0, 1],
        ...     [1, 1, 1, 0, 0],
        ...     [0, 1, 1, 1, 0],
        ...     [0, 0, 1, 1, 1],
        ...     [1, 0, 0, 1, 1]
        ... ])
        >>> G = NoncommutativeGraph.from_adjmat(adj_mat)
        >>> theta = G.lovasz_theta()
        >>> abs(theta - numpy.sqrt(5)) < 1e-8
        True
        """

        assert len(adj_mat.shape) == 2
        assert adj_mat.shape[0] == adj_mat.shape[1]
        assert np.all(adj_mat == adj_mat.transpose())
        n = adj_mat.shape[0]
        basis = []

        # copy and cast to numpy
        adj_mat = np.array(adj_mat)

        for (i, j) in np.transpose(adj_mat.nonzero()):
            m = np.zeros((n, n), dtype=complex)
            m[i, j] = 1
            basis.append(m)

        return cls(TensorSubspace.from_span(basis))

    @classmethod
    def from_sagegraph(cls, G):
        """
        Create a non-commutative graph from a Sage Graph.

        Actually, all that is required is that the input G supports an
        adjacency_matrix method.
        """

        return cls.from_adjmat(G.adjacency_matrix())

    @classmethod
    def pentagon(cls):
        """
        Create the 5-cycle graph.  Useful for testing.
        """

        # Adjacency matric for the 5-cycle graph.
        adj_mat = np.array([
            [1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1]
        ])
        G = cls.from_adjmat(adj_mat)
        return G

    @classmethod
    def random(cls, spc, num_seeds):
        if isinstance(spc, HilbertSpace):
            spc.assert_ket_space()
            n = spc.dim()
        else:
            assert isinstance(spc, int)
            n = spc
            spc = None

        S = TensorSubspace.from_span([ np.eye(n, dtype=complex) ])
        for i in range(num_seeds):
            M = np.random.standard_normal(size=(n,n)) + \
                np.random.standard_normal(size=(n,n))*1j
            S |= TensorSubspace.from_span([ M + M.conj().T ])

        if spc is not None:
            S = S.map(lambda x: spc.O.array(x, reshape=True))

        return NoncommutativeGraph(S)

    def _get_S_ot_L_basis(self, Sb):
        """
        Compute a basis for the allowed Y operators for Theorem 9 of
        arXiv:1002.2514.  These are the operators which are Hermitian and are
        in S*L(A').  Note that this basis is intended to map real vectors to
        complex Y operators.
        """

        (nS, n, _n) = Sb.shape
        assert n == _n

        Lb = TensorSubspace.full((n, n)).hermitian_basis()

        baz = np.zeros((nS*n*n, n, n, n, n), dtype=complex)
        i = 0
        for x in Sb:
            for y in Lb:
                baz[i] = np.tensordot(x, y, axes=([],[])).transpose((0, 2, 1, 3))
                i += 1
        assert i == baz.shape[0]

        ret = baz.transpose((1, 2, 3, 4, 0))

        # [ |a>, |a'>, <a|, <a'| ; idx ]
        return ret

    def _basis_doubly_hermit(self, Sb):
        """
        Returns a basis of elements of spc \ot spc that are Hermitian and also have Hermitian
        images under R().

        Sb must be a Hermitian basis.
        """

        (nS, n, _n) = Sb.shape
        assert n == _n

        if nS == 0:
            return np.zeros((0, n, n, n, n), dtype=complex)

        out = []
        for (i, x) in enumerate(Sb):
            out.append( np.tensordot(x, x.conj(), axes=([],[])).transpose((0, 2, 1, 3)) )
            for (j, y) in enumerate(Sb):
                if j >= i:
                    continue
                xy = np.tensordot(x, y.conj(), axes=([],[])).transpose((0, 2, 1, 3))
                yx = np.tensordot(y, x.conj(), axes=([],[])).transpose((0, 2, 1, 3))
                out.append(xy+yx)

        out = [ x / linalg.norm(x) for x in out ]

        ret = np.array(out).transpose((1, 2, 3, 4, 0))

        # [ |a>, |a'>, <a|, <a'| ; idx ]
        return ret

    def _test_get_Y_basis_doubly_hermit(self):
        n = self.n
        Yb = self.Y_basis
        Hb = self.full_basis_dh
        A = TensorSubspace.from_span([ mat_cplx_to_real(Yb[:,:,:,:,i].reshape((n*n, n*n))) for i in range(Yb.shape[4]) ])
        B = TensorSubspace.from_span([ mat_cplx_to_real(Hb[:,:,:,:,i].reshape((n*n, n*n))) for i in range(Hb.shape[4]) ])
        C = A & B
        print(A,B,C)
        out = np.array([ mat_real_to_cplx(x).reshape((n,n,n,n)) for x in C ])
        for (i,c) in enumerate(np.rollaxis(Hb, -1)):
            x = c.reshape((n*n, n*n))
            y = mat_real_to_cplx(mat_cplx_to_real(x))
            assert linalg.norm(x - x.conj().T) < 1e-10
            assert np.allclose(x, y)
        for (i,c) in enumerate(B):
            x = mat_real_to_cplx(c)
            assert linalg.norm(x - x.conj().T) < 1e-10
        for (i,c) in enumerate(C):
            x = mat_real_to_cplx(c)
            assert linalg.norm(x - x.conj().T) < 1e-10
        a = TensorSubspace.from_span([ mat_cplx_to_real(x.reshape(n*n,n*n)) for x in
            np.rollaxis(self.Y_basis_dh, -1) ])
        b = TensorSubspace.from_span([ mat_cplx_to_real(x.reshape(n*n,n*n)) for x in out ])
        print(a)
        print(b)
        print(a.equiv(b))
        assert a.equiv(b)

    def _test_doubly_hermitian_basis(self):
        n = self.n

        def perms(i,j,k,l):
            return [(i,j,k,l), (j,i,l,k), (l,k,j,i), (k,l,i,j)]

        inds = set()
        for (i,j,k,l) in itertools.product(list(range(n)), repeat=4):
            p = perms(i,j,k,l)
            if not np.any([ x in inds for x in p ]):
                inds.add((i,j,k,l))

        ops = []
        for (i,j,k,l) in inds:
            a = np.zeros((n,n,n,n), dtype=complex)
            a[i,j,k,l] = 1
            a += a.transpose((1,0,3,2))
            a += a.transpose((2,3,0,1))
            ops.append(a)

            a = np.zeros((n,n,n,n), dtype=complex)
            a[i,j,k,l] = 1j
            a += a.transpose((1,0,3,2)).conj()
            a += a.transpose((2,3,0,1)).conj()
            if np.sum(np.abs(a)) > 1e-6:
                ops.append(a)

        a = TensorSubspace.from_span([ mat_cplx_to_real(x.reshape(n*n,n*n)) for x in
            np.rollaxis(self.full_basis_dh, -1) ])
        b = TensorSubspace.from_span([ mat_cplx_to_real(x.reshape(n*n,n*n)) for x in ops ])
        print(a)
        print(b)
        print(a.equiv(b))
        assert a.equiv(b)

    def lovasz_theta(self, long_return=False):
        """
        Compute the non-commutative generalization of the Lovasz function,
        using Theorem 9 of arXiv:1002.2514.

        If the long_return option is True, then some extra status and internal
        quantities are returned (such as the optimal Y operator).
        """

        return self.unified_dual(self.Y_basis, [], [], long_return)

    def szegedy(self, cone, long_return=False):
        """
        My non-commutative generalization of Szegedy's number.

        # FIXME - make it match the paper (primal and dual)
        min t s.t.
            tI - Tr_A Y \succeq 0
            Y \in S \ot \mathcal{L}
            Y \succeq \Phi
            R(Y) \succeq 0
            optional: R(Y) \in cone

        If the long_return option is True, then some extra status and internal
        quantities are returned (such as the optimal Y operator).
        """

        v = {
            False: [self.cond_psd],
            True:  [self.cond_psd, self.cond_ppt],
            'psd': [self.cond_psd],
            'psd&ppt': [self.cond_psd, self.cond_ppt],
            'ppt': [self.cond_ppt],
        }[cone]

        return self.unified_dual(self.Y_basis_dh, v, [], long_return)

    def get_five_values(self):
        """
        >>> np.random.seed(1)
        >>> S = NoncommutativeGraph.random(3, 5)
        >>> cvxopt.solvers.options['abstol'] = float(1e-7)
        >>> cvxopt.solvers.options['reltol'] = float(1e-7)
        >>> vals = S.get_five_values()
        >>> good = np.array([1.0000000895503431, 2.5283253619689754, 2.977455214593435, \
                    2.9997454690478249, 2.9999999897950529])
        >>> np.sum(np.abs(vals-good)) < 1e-5
        True
        """

        (nS, n, _n) = self.S_basis.shape
        assert n == _n

        return [
            self.schrijver(True),
            self.schrijver(False),
            self.lovasz_theta(),
            self.szegedy(False),
            self.szegedy(True),
        ]

    def unified_dual(self, Y_basis, extra_constraints, extra_vars, long_return=False):
        """
        Compute Lovasz/Schrijver/Szegedy type quantities.

        min t s.t.
            tI - Tr_A (Y-Z) \succeq 0
            Y \in S \ot \mathcal{L}
            Y-Z \succeq \Phi
            R(Z) \in \sum( extra_vars )
            R(Y) \in \cap( extra_constraints )
        """

        n = self.n

        Yb_len = Y_basis.shape[4]

        # x = [t, Y.A:Si * Y.A':i * Y.A':j, Z]
        xvec_len = 1 + Yb_len + np.sum([ v['basis'].shape[-1] for v in extra_vars], dtype=int)

        idx = 1
        x_to_Y = np.zeros((n,n,n,n,xvec_len), dtype=complex)
        x_to_Y[:,:,:,:,idx:idx+Yb_len] = Y_basis
        idx += Yb_len

        x_to_Z = []
        for v in extra_vars:
            xZ = np.zeros((n,n,n,n,xvec_len), dtype=complex)
            Zb_len = v['basis'].shape[-1]
            xZ[:,:,:,:,idx:idx+Zb_len] = v['basis']
            idx += Zb_len
            x_to_Z.append(xZ)

        assert idx == xvec_len

        phi_phi = np.zeros((n,n, n,n), dtype=complex)
        for (i, j) in itertools.product(list(range(n)), repeat=2):
            phi_phi[i, i, j, j] = 1
        phi_phi = phi_phi.reshape(n**2, n**2)

        # Cost vector.
        # x = [t, Y.A:Si * Y.A':i * Y.A':j]
        c = np.zeros(xvec_len)
        c[0] = 1

        # tI - tr_A{Y-Z} >= 0
        Fx_1 = -np.trace(x_to_Y, axis1=0, axis2=2)
        for xZ in x_to_Z:
            Fx_1 += np.trace(xZ, axis1=0, axis2=2)
        for i in range(n):
            Fx_1[i, i, 0] = 1
        F0_1 = np.zeros((n, n))

        # Y - Z  >=  |phi><phi|
        Fx_2 = x_to_Y.reshape(n**2, n**2, xvec_len).copy()
        for xZ in x_to_Z:
            Fx_2 -= xZ.reshape(n**2, n**2, xvec_len)
        F0_2 = phi_phi

        Fx_evars = []
        F0_evars = []
        for (xZ, v) in zip(x_to_Z, extra_vars):
            Fx = np.array([ v['R'](z) for z in np.rollaxis(xZ, -1) ], dtype=complex)
            Fx = np.rollaxis(Fx, 0, len(Fx.shape))
            F0 = v['0']
            Fx_evars.append(Fx)
            F0_evars.append(F0)

        Fx_econs = []
        F0_econs = []
        for v in extra_constraints:
            Fx = np.array([ v['R'](y) for y in np.rollaxis(x_to_Y, -1) ], dtype=complex)
            Fx = np.rollaxis(Fx, 0, len(Fx.shape))
            F0 = v['0']
            Fx_econs.append(Fx)
            F0_econs.append(F0)

        Fx_list = [Fx_1, Fx_2] + Fx_evars + Fx_econs
        F0_list = [F0_1, F0_2] + F0_evars + F0_econs

        (xvec, sdp_stats) = call_sdp(c, Fx_list, F0_list)

        if sdp_stats['status'] in ['optimal', 'primal infeasible']:
            rho = mat_real_to_cplx(np.array(sdp_stats['zs'][0]))
            I_ot_rho = np.tensordot(np.eye(n), rho, axes=0).transpose(0,2,1,3)
            zs1 = mat_real_to_cplx(np.array(sdp_stats['zs'][1])).reshape(n,n,n,n)
            # FIXME - aren't Fx_evars before Fx_econs?
            L_list = []
            for (i,v) in enumerate(extra_constraints):
                zsi = mat_real_to_cplx(np.array(sdp_stats['zs'][2+i]))
                L_list.append(v['R*'](zsi))
            L_sum = np.sum(L_list, axis=0)
            T = zs1 - I_ot_rho

            # Verify dual solution (or part of it; more is done below)
            verify_tol=1e-7
            if verify_tol:
                # Test the primal solution
                for mat in np.rollaxis(Y_basis, -1):
                    dp = np.tensordot(T+L_sum, mat.conj(), axes=4)
                    err = linalg.norm(dp)
                    if err > verify_tol: print("WARNING T+L_sum in Y_basis.perp() err =", err)
                T_plus_Irho = T + I_ot_rho
                err = check_psd(T_plus_Irho.reshape(n**2, n**2))
                if err < -verify_tol: print("WARNING: T_plus_Irho pos err =", err)
                err = abs(np.trace(rho) - 1)
                if err < -verify_tol: print("WARNING: Tr(rho) err =", err)
                err = check_psd(rho)
                if err < -verify_tol: print("WARNING: rho pos err =", err)

                for (i, (v, L)) in enumerate(zip(extra_constraints, L_list)):
                    M = v['R'](L) - v['0']
                    err = check_psd(M)
                    if err < -verify_tol: print("WARNING: R(L_%d) err = %g" % (i, err))

                # FIXME - doesn't pass
                for (i, v) in enumerate(extra_vars):
                    M = v['R'](T) - v['0']
                    print('Herm:', linalg.norm(M - M.conj().T))
                    err = check_psd(M)
                    if err < -verify_tol: print("WARNING: R(T) [%d] err = %g" % (i, err))

        if sdp_stats['status'] == 'optimal':
            t = xvec[0]
            Y = np.dot(x_to_Y, xvec)
            Z_list = [ np.dot(xZ, xvec) for xZ in x_to_Z ]
            Z_sum = np.sum(Z_list, axis=0)

            # Verify primal/dual solution
            verify_tol=1e-7
            if verify_tol:
                err = abs(T.trace().trace() + 1 - t)
                if err > verify_tol: print("WARNING: primal vs dual err =", err)

                err = check_psd((Y-Z_sum).reshape(n**2, n**2) - phi_phi)
                if err < -verify_tol: print("WARNING: phi_phi err =", err)

                for (i, (v, Z)) in enumerate(zip(extra_vars, Z_list)):
                    M = v['R'](Z) - v['0']
                    err = check_psd(M)
                    if err < -verify_tol: print("WARNING: R(Z_%d) err = %g" % (i, err))

                for (i, v) in enumerate(extra_constraints):
                    M = v['R'](Y) - v['0']
                    err = check_psd(M)
                    if err < -verify_tol: print("WARNING: R(Y) [%d] err = %g" %(i, err))

                maxeig = linalg.eigvalsh(np.trace(Y-Z_sum, axis1=0, axis2=2))[-1].real
                err = abs(xvec[0] - maxeig)
                if err > verify_tol: print("WARNING: t err =", err)

                # make sure it is in S*L(A')
                for mat in self.Sp_basis:
                    dp = np.tensordot(Y, mat.conj(), axes=[[0, 2], [0, 1]])
                    err = linalg.norm(dp)
                    if err > verify_tol: print("WARNING: Y in S \ot L(A') err =", err)

            if long_return:
                ret = {}
                for key in [
                    'n', 'x_to_Y', 'x_to_Z',
                    'phi_phi', 'c', 't', 'Y', 'Z_list', 'xvec', 'sdp_stats',
                    'T', 'rho', 'T_plus_Irho', 'L_list', 'L_sum',
                ]:
                    ret[key] = locals()[key]
                return ret
            else:
                return t
        elif sdp_stats['status'] == 'primal infeasible':
            t = np.inf
            if long_return:
                ret = {}
                for key in [
                    'n', 'x_to_Y', 'x_to_Z',
                    'phi_phi', 'c', 't', 'xvec', 'sdp_stats',
                    'T', 'rho', 'T_plus_Irho', 'L_list', 'L_sum',
                ]:
                    ret[key] = locals()[key]
                return ret
            else:
                return t
        else:
            raise Exception('cvxopt.sdp returned error: '+sdp_stats['status'])

    def schrijver(self, cones, long_return=False):
        r"""
        My non-commutative generalization of Schrijver's number.

        max <\Phi|T + I \ot \rho|\Phi> s.t.
            \rho \succeq 0, \Tr(\rho)=1
            T + I \ot \rho \succeq 0
            T \in S^\perp \ot S^\perp
            T^\ddag = T
            R(T) \in cones

        min ||Tr_A(Y)|| s.t.
            Y \succeq |\Phi><\Phi|
            Y+L-X \in S \djp S
            R(L) \in cones^*
            X^\ddag = -X
            X^\dag = X

        If the long_return option is True, then some extra status and internal
        quantities are returned (such as the optimal Y operator).
        """

        if not isinstance(cones, list):
            assert isinstance(cones, str)
            cones = {
                'hermit': [],
                'psd': [self.cond_psd],
                'psd&ppt': [self.cond_psd, self.cond_ppt],
                'ppt': [self.cond_ppt],
            }[cones]

        for C in cones:
            assert 'R' in C

        n = self.n

        Tbas = self.T_basis_dh
        Tb_len = Tbas.shape[4]

        # rhotf is the trace-free component of the actual rho
        rhotf_basis = TensorSubspace.from_span([np.eye(n)]).perp(). \
            hermitian_basis().transpose((1,2,0))
        rb_len = rhotf_basis.shape[2]
        assert rb_len == n*n-1

        xvec_len = Tb_len + rb_len

        idx = 0
        x_to_T = np.zeros((n,n,n,n,xvec_len), dtype=complex)
        x_to_T[:,:,:,:,idx:idx+Tb_len] = Tbas
        idx += Tb_len

        x_to_rhotf = np.zeros((n,n,xvec_len), dtype=complex)
        x_to_rhotf[:,:,idx:idx+rb_len] = rhotf_basis
        idx += rb_len

        assert idx == xvec_len

        # T + I \ot rhotf
        x_to_sum = x_to_T + \
                np.tensordot(np.eye(n), x_to_rhotf, axes=0).transpose((0,2,1,3,4))

        # rho \succeq 0
        Fx_1 = x_to_rhotf
        F0_1 = -np.eye(n)/n

        # T + I \ot rho \succeq 0
        Fx_2 = x_to_sum.reshape(n**2, n**2, xvec_len)
        for i in range(Fx_2.shape[2]):
            assert linalg.norm(Fx_2[:,:,i] - Fx_2[:,:,i].conj().T) < 1e-10
        F0_2 = -np.eye(n**2)/n

        c = -np.trace(np.trace(x_to_sum)).real

        Fx_econs = []
        F0_econs = []
        for v in cones:
            Fx = np.array([ v['R'](y) for y in np.rollaxis(x_to_T, -1) ], dtype=complex)
            Fx = np.rollaxis(Fx, 0, len(Fx.shape))
            F0 = v['0']
            Fx_econs.append(Fx)
            F0_econs.append(F0)

        Fx_list = [Fx_1, Fx_2] + Fx_econs
        F0_list = [F0_1, F0_2] + F0_econs

        (xvec, sdp_stats) = call_sdp(c, Fx_list, F0_list)

        if sdp_stats['status'] == 'optimal':
            t = -np.dot(c, xvec) + 1
            T = np.dot(x_to_T, xvec)
            rho = np.dot(x_to_rhotf, xvec) + np.eye(n)/n
            I_rho = np.tensordot(np.eye(n), rho, axes=0).transpose(0,2,1,3)
            T_plus_Irho = np.dot(x_to_sum, xvec) + np.eye(n*n).reshape(n,n,n,n) / n

            J = np.zeros((n,n, n,n), dtype=complex)
            for (i, j) in itertools.product(list(range(n)), repeat=2):
                J[i, i, j, j] = 1

            # FIXME - zs0 appears to always be zero.  But is it ever needed for constructing
            # the dual solution?
            zs0 = mat_real_to_cplx(np.array(sdp_stats['zs'][0]))
            Y = J + mat_real_to_cplx(np.array(sdp_stats['zs'][1])).reshape(n,n,n,n)
            zs_idx = 2

            L_list = []
            for (i,v) in enumerate(cones):
                zsi = mat_real_to_cplx(np.array(sdp_stats['zs'][zs_idx]))
                zs_idx += 1
                L_list.append(v['R*'](zsi))
            # FIXME - redefined below
            L_sum = np.sum(L_list, axis=0)

            assert zs_idx == len(sdp_stats['zs'])

            # cvxopt guarantees the dual to give zero here
            #foo = (
            #    np.tensordot(zs0, x_to_rhotf.conj(), axes=2) +
            #    np.tensordot(Y, x_to_sum.conj(), axes=4)
            #)
            #if len(L_list):
            #    foo += np.tensordot(L_sum, x_to_T.conj(), axes=4)
            #print('**', linalg.norm(foo))

            # Extract rot-antihermit portion of L and put it in Y.
            # Copy rot-antihermit portion of Y to X.
            Ldh_list = [ project_dh(L) for L in L_list ]
            Y += np.sum(Ldh_list, axis=0) - np.sum(L_list, axis=0)
            L_list = Ldh_list
            L_sum = np.sum(L_list, axis=0)
            X = Y - project_dh(Y)

            # FIXME - is X guaranteed Hermitian?
            # Should I require the cones to be doubly hermit?
            print(linalg.norm(X - X.transpose(2,3,0,1).conj()))

            verify_tol=1e-7
            if verify_tol:
                # Test the primal solution
                err = linalg.norm(T + I_rho - T_plus_Irho)
                if err > verify_tol: print("WARNING: T + I \ot rho err =", err)
                err = abs(t - T_plus_Irho.trace(axis1=0, axis2=1).trace(axis1=0, axis2=1))
                if err > verify_tol: print("WARNING: primal value err =", err)

                for mat in self.S_basis:
                    dp = np.tensordot(T, mat.conj(), axes=[[0, 2], [0, 1]])
                    err = linalg.norm(dp)
                    if err > verify_tol: print("WARNING: T in S^\perp \ot S^\perp err =", err)

                err = check_psd(T_plus_Irho.reshape(n**2, n**2))
                if err < -verify_tol: print("WARNING: T_plus_Irho pos err =", err)
                err = abs(np.trace(rho) - 1)
                if err < -verify_tol: print("WARNING: Tr(rho) err =", err)
                err = check_psd(rho)
                if err < -verify_tol: print("WARNING: rho pos err =", err)

                for (i, v) in enumerate(cones):
                    M = v['R'](T) - v['0']
                    err = check_psd(M)
                    if err < -verify_tol: print("WARNING: R(T) [%d] err = %g" % (i, err))

                # Test the dual solution
                err_Y_space = 0
                for matA in self.Sp_basis:
                    for matB in self.Sp_basis:
                        xy = np.tensordot(matA, matB.conj(), axes=([],[])).transpose((0, 2, 1, 3))
                        dp = np.tensordot(Y+L_sum-X, xy.conj(), axes=4)
                        err_Y_space += abs(dp)
                if err_Y_space > verify_tol: print("WARNING: Y+L-X in S \djp S err =", err_Y_space)

                for (i, (v, L)) in enumerate(zip(cones, L_list)):
                    M = v['R'](L) - v['0']
                    err = check_psd(M)
                    if err < -verify_tol: print("WARNING: R(L%d) err = %g" % (i, err))

                err = check_psd((Y-J).reshape(n*n, n*n))
                if err < -verify_tol: print("WARNING: Y-J pos err =", err)

                err = abs(t - linalg.eigvalsh(Y.trace(axis1=0, axis2=2))[-1])
                if err > verify_tol: print("WARNING: dual value err =", err)

            if long_return:
                if self.S._hilb_space is not None:
                    ha = self.S._hilb_space.ket_space()
                    hb = ha.prime # FIXME
                rho = hb.O.array(rho)
                T = (ha*hb).O.array(T)
                Y = (ha*hb).O.array(Y)
                L_map = { C['name']: (ha*hb).O.array(L) for (C, L) in zip(cones, L_list) }
                #L_sum = (ha*hb).O.array(L_sum)
                X = (ha*hb).O.array(X)
                #return locals()
                ret = {}
                for key in [
                    't', 'T', 'rho', 'Y', 'L_map', 'X', 'ha', 'hb'
                ]:
                    ret[key] = locals()[key]
                return ret
            else:
                return t
        else:
            raise Exception('cvxopt.sdp returned error: '+sdp_stats['status'])

    def make_channel(self):
        """
        Makes a CPTP map whose confusibility graph is equal to `self`.
        """

        S = self.S
        assert S._hilb_space is not None
        spc = S._hilb_space
        assert S.is_hermitian()
        assert spc.eye() in S

        if S.dim() == 1:
            return CP_Map.identity(spc)

        B = (S - spc.eye().span()).hermitian_basis()
        B = [ b - spc.eye() * b.eigvalsh()[0] for b in B ]
        m = np.sum(B).eigvalsh()[-1]
        B = [ b/m for b in B ]
        B += [ spc.eye() - np.sum(B) ]
        J = [ b.sqrt() for b in B ]
        J = [ j.svd()[0].H * j for j in J ]

        hk = qudit('k', len(J))
        J = [ hk.ket(k) * j for (k,j) in enumerate(J) ]

        Kspc = TensorSubspace.from_span(J)
        assert S.equiv(Kspc.H * Kspc)

        chan = CP_Map.from_kraus(J)
        assert chan.is_cptp()

        return chan

### Validation code ####################

def test_schrijver():
    ha = qudit('a', 3)
    np.random.seed(2)
    G = NoncommutativeGraph.random(ha, 3)

    cvxopt.solvers.options['abstol'] = float(1e-7)
    cvxopt.solvers.options['reltol'] = float(1e-7)

    info = G.schrijver('psd&ppt', True)
    ret = schrijver_feasibility(G, frozenset(['psd','ppt']), *[ info[x] for x in 'ha,hb,t,rho,T,Y,L_map,X'.split(',') ])
    return ret

def schrijver_feasibility(G, used_cones, ha, hb, t, rho, T, Y, L_map, X):
    r"""
    Verify Schrijver solution.

    t = max <\Phi|T + I \ot \rho|\Phi> s.t.
        \rho \succeq 0, \Tr(\rho)=1
        T + I \ot \rho \succeq 0
        T \in S^\perp \ot S^\perp
        T^\ddag = T
        R(T) \in cones

    t = min ||Tr_A(Y)|| s.t.
        Y \succeq |\Phi><\Phi|
        Y+L-X \in S \djp S
        R(L) \in cones^*
        X^\ddag = -X
        X^\dag = X
    """

    S = G.S
    err = {}

    assert S._hilb_space == ha.O
    assert rho.space == hb.O
    assert T.space == (ha*hb).O
    assert Y.space == (ha*hb).O
    for L in L_map.values():
        assert L.space == (ha*hb).O
    assert X.space == (ha*hb).O
    assert L_map.keys() == used_cones

    def R(x):
        return x.relabel({ hb: ha.H, ha.H: hb })

    def ddag(x):
        ret = x.relabel({ ha: hb, hb: ha, ha.H: hb.H, hb.H: ha.H }).conj()
        assert (ret - R(R(x).H)).norm() < 1e-12
        return ret

    Phi = ha.eye().relabel({ ha.H: hb })
    J = Phi.O

    ### Verify primal

    err[r'primal val'] = abs(t - (1 + Phi.H * T * Phi))
    err[r'trace(rho)'] = abs(1 - rho.trace())
    err[r'rho PSD'] = check_psd(rho)
    err[r'T + I \ot rho PSD'] = check_psd(T + ha.eye()*rho)
    err[r'T^\ddag - T'] = (T - ddag(T)).norm()

    for C in used_cones:
        if C == 'psd':
            err[r'T_PSD'] = check_psd(R(T))
        elif C == 'ppt':
            err[r'T_PPT'] = check_psd(R(T).transpose(ha))
        else:
            assert 0

    # FIXME - T \in S^\perp \ot S^\perp

    ### Verify dual

    err[r'dual val'] = abs(t - Y.trace(ha).eigvalsh()[-1])
    err[r'Y \succeq J'] = check_psd(Y - J)

    # FIXME - Y+L-X \in S \djp S

    for C in used_cones:
        L = L_map[C]
        if C == 'psd':
            err[r'L_PSD'] = check_psd(R(L))
        elif C == 'ppt':
            err[r'L_PPT'] = check_psd(R(L).transpose(ha))
        else:
            assert 0

    err[r'X^\ddag + X'] = (X + ddag(X)).norm()
    err[r'X^\dag - X'] = (X - X.H).norm()

    ### Tally and report

    assert min(err.values()) >= 0

    for (k, v) in err.items():
        if v > 1e-7:
            print('err[%s] = %g' % (k, v))

    print('total err:', sum(err.values()))

    # FIXME
    return locals()

#if __name__ == "__main__":
#    cvxopt.solvers.options['show_progress'] = False
#    # Unfortunately, Schrijver doesn't converge well.
#    cvxopt.solvers.options['abstol'] = float(1e-5)
#
#    S = NoncommutativeGraph.random(3, 3)
#    print(S)
#
#    vals = S.get_five_values()
#    for v in vals:
#        print(v)

# If this module is run from the command line, run the doctests.
if __name__ == "__main__":
    # Doctests require not getting progress messages from SDP solver.
    cvxopt.solvers.options['show_progress'] = False

# FIXME
#    print("Running doctests.")
#
#    import doctest
#    doctest.testmod()

    locals().update(test_schrijver())

    # seed=5 NoncommutativeGraph.random(4, 5) gives gap between Szegedy with positive and with
    # just Hermitian.
    #cvxopt.solvers.options['abstol'] = float(1e-7)
    #cvxopt.solvers.options['reltol'] = float(1e-7)
    #cvxopt.solvers.options['show_progress'] = True

    #print('th dual:  ', S.lovasz_theta())
    #a = S.unified_primal(S.T_basis, [], [], True)
    #print('th primal:', a['t'])

    #a = S.szegedy('psd&ppt', long_return=True)
    #print('thp dual:  ', a['t'])
    #print('---------')
    #b = S.unified_primal(S.T_basis, [], [S.cond_psd], True)
    #print('thp primal:', b['t'])

    #a = S.schrijver('psd&ppt', long_return=True)
    #print('thm dual:  ', a['t'])
    #print('---------')
    #b = S.schrijver([S.cond_psd, S.cond_ppt], True)
    #print('---------')
    #b = S.schrijver([S.cond_ppt], True)
    #print('---------')
    #b = S.schrijver([S.cond_psd], True)
    #print('---------')
    #print('thm primal:', b['t'])

    #locals().update(b)

    #zG_list = []
    #for (z, G) in zip(sdp_stats['zs'], sdp_stats['Gs']):
    #    v = np.array(z).reshape(z.size[0]**2)
    #    zG = np.dot(np.array(G).T, v)
    #    zG_list.append(zG)
    #print(linalg.norm(np.sum(zG_list, axis=0) + c))

    #t_honly = S.unified_dual(S.Y_basis_dh, [], [], False)
    #print('hermit only:', t_honly)

    #b = S.szegedy(True, long_return=True)
    ##b = S.lovasz_theta(long_return=True)
    #locals().update(b)
    #print(t)

    #zs1 = mat_real_to_cplx(np.array(sdp_stats['zs'][1])).reshape(n,n,n,n) * 2
    #zs2 = mat_real_to_cplx(np.array(sdp_stats['zs'][2])).reshape(n,n,n,n) * 2
    ##zs3 = mat_real_to_cplx(np.array(sdp_stats['zs'][3])).reshape(n,n,n,n) * 2
    #print(T.trace().trace()+1 - t)
    #w=[np.tensordot(x, y.conj(), axes=([],[])).transpose((0, 2, 1, 3)) for x in S.S_basis for y in S.S_basis]
    #print(np.max([linalg.norm(np.tensordot(T, (x + x.transpose(1,0,3,2).conj()).conj(), axes=4)) for x in w]))
