# Noncommutative graphs as defined by Duan, Severini, Winter in arXiv:1002.2514.

import numpy as np
import numpy.linalg as linalg
import itertools
import cvxopt.base
import cvxopt.solvers

from qitensor.subspace import TensorSubspace

# This is the only thing that is exported.
__all__ = ['NoncommutativeGraph']

### Some helper functions for cvxopt ###

def mat_cplx_to_real(cmat):
    rmat = np.zeros((2, cmat.shape[0], 2, cmat.shape[1]))
    rmat[0, :, 0, :] = cmat.real
    rmat[1, :, 1, :] = cmat.real
    rmat[0, :, 1, :] = -cmat.imag
    rmat[1, :, 0, :] = cmat.imag
    # preserve the norm
    rmat /= np.sqrt(2)
    return rmat.reshape(cmat.shape[0]*2, cmat.shape[1]*2)

# This could help for extracting the dual solution for the solver.  But I haven't yet figured
# out how to interpret the things that cvxopt returns.
#ret=NoncommutativeGraph(S).lovasz_theta(long_return=True)
#ss=mat_real_to_cplx(np.array(ret['sdp_stats']['ss'][1]))
#zs=mat_real_to_cplx(np.array(ret['sdp_stats']['zs'][1]))
def mat_real_to_cplx(rmat):
    w = rmat.shape[0]/2
    h = rmat.shape[1]/2
    return (rmat[:w,:h] + 1j*rmat[w:,:h]) * np.sqrt(2)

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
    (Fx_list, F0_list) = make_F_real(Fx_list, F0_list)
    Gs = [cvxopt.base.matrix(-Fx.reshape(Fx.shape[0]**2, Fx.shape[2])) for Fx in Fx_list]
    hs = [cvxopt.base.matrix(-F0) for F0 in F0_list]

    sol = cvxopt.solvers.sdp(cvxopt.base.matrix(c), Gs=Gs, hs=hs)
    xvec = np.array(sol['x']).flatten()

    if sol['status'] == 'optimal':
        for (G, h) in zip(Gs, hs):
            G = np.array(G)
            h = np.array(h)
            M = np.dot(G, xvec).reshape(h.shape)
            assert linalg.eigvalsh(h-M)[0] > -1e-7

    return (xvec, sol)

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
            'basis': self.full_basis_dh,
            'R':  lambda Z: Z.transpose((0,2,1,3)).reshape(n**2, n**2),
            'R*': lambda Z: Z.reshape(n,n,n,n).transpose((0,2,1,3)),
            '0': np.zeros((n**2, n**2), dtype=complex),
        }

        self.cond_ppt = {
            'basis': self.full_basis_dh,
            'R':  lambda Z: Z.transpose((1,2,0,3)).reshape(n**2, n**2),
            'R*': lambda Z: Z.reshape(n,n,n,n).transpose((2,0,1,3)),
            '0': np.zeros((n**2, n**2), dtype=complex),
        }

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
    def random(cls, n, num_seeds):
        S = TensorSubspace.from_span([ np.eye(n, dtype=complex) ])
        for i in range(num_seeds):
            M = np.random.standard_normal(size=(n,n)) + \
                np.random.standard_normal(size=(n,n))*1j
            S |= TensorSubspace.from_span([ M + M.conj().T ])
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

        out = []
        for (i, x) in enumerate(Sb):
            out.append( np.tensordot(x, x.conj(), axes=([],[])).transpose((0, 2, 1, 3)) )
            for (j, y) in enumerate(Sb):
                if j >= i:
                    continue;
                xy = np.tensordot(x, y.conj(), axes=([],[])).transpose((0, 2, 1, 3))
                yx = np.tensordot(y, x.conj(), axes=([],[])).transpose((0, 2, 1, 3))
                out.append(xy+yx)

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
        print A,B,C
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
        a = TensorSubspace.from_span([ mat_cplx_to_real(x.reshape(n*n,n*n)) for x in \
                np.rollaxis(self.Y_basis_dh, -1) ])
        b = TensorSubspace.from_span([ mat_cplx_to_real(x.reshape(n*n,n*n)) for x in out ])
        print a
        print b
        print a.equiv(b)
        assert a.equiv(b)

    def _test_doubly_hermitian_basis(self):
        n = self.n

        def perms(i,j,k,l):
            return [(i,j,k,l), (j,i,l,k), (l,k,j,i), (k,l,i,j)]

        inds = set()
        for (i,j,k,l) in itertools.product(range(n), repeat=4):
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

        a = TensorSubspace.from_span([ mat_cplx_to_real(x.reshape(n*n,n*n)) for x in \
                np.rollaxis(self.full_basis_dh, -1) ])
        b = TensorSubspace.from_span([ mat_cplx_to_real(x.reshape(n*n,n*n)) for x in ops ])
        print a
        print b
        print a.equiv(b)
        assert a.equiv(b)

    def lovasz_theta(self, long_return=False):
        """
        Compute the non-commutative generalization of the Lovasz function,
        using Theorem 9 of arXiv:1002.2514.

        If the long_return option is True, then some extra status and internal
        quantities are returned (such as the optimal Y operator).
        """

        return self.unified_dual(self.Y_basis, [], [], long_return)

    def schrijver(self, ppt, long_return=False):
        """
        My non-commutative generalization of Schrijver's number.

        min t s.t.
            tI - Tr_A (Y-Z) \succeq 0
            Y \in S \ot \mathcal{L}
            Y-Z (-Z2) \succeq \Phi
            R(Z) \succeq 0
            optional: R(Z2) \in PPT

        If the long_return option is True, then some extra status and internal
        quantities are returned (such as the optimal Y operator).
        """

        v = [self.cond_psd]
        if ppt:
            v.append(self.cond_ppt)
        return self.unified_dual(self.Y_basis, [], v, long_return)

    def szegedy(self, ppt, long_return=False):
        """
        My non-commutative generalization of Schrijver's number.

        min t s.t.
            tI - Tr_A Y \succeq 0
            Y \in S \ot \mathcal{L}
            Y \succeq \Phi
            R(Y) \succeq 0
            optional: R(Y) \in PPT

        If the long_return option is True, then some extra status and internal
        quantities are returned (such as the optimal Y operator).
        """

        v = [self.cond_psd]
        if ppt:
            v.append(self.cond_ppt)
        return self.unified_dual(self.Y_basis_dh, v, [], long_return)


    def get_five_values(self):
        """
        >>> np.random.seed(1)
        >>> S = NoncommutativeGraph.random(3, 5)
        >>> # FIXME - Unfortunately, Schrijver doesn't converge well.
        >>> cvxopt.solvers.options['abstol'] = float(1e-5)
        >>> cvxopt.solvers.options['reltol'] = float(1e-5)
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
        xvec_len = 1 + Yb_len + np.sum([ v['basis'].shape[-1] for v in extra_vars])

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
        for (i, j) in itertools.product(range(n), repeat=2):
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
        for i in xrange(n):
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

        if sdp_stats['status'] == 'optimal':
            t = xvec[0]
            Y = np.dot(x_to_Y, xvec)
            Z_list = [ np.dot(xZ, xvec) for xZ in x_to_Z ]
            Z_sum = np.sum(Z_list, axis=0)

            rho = mat_real_to_cplx(np.array(sdp_stats['zs'][0]))
            I_ot_rho = np.tensordot(np.eye(n), rho, axes=0).transpose(0,2,1,3)
            zs1 = mat_real_to_cplx(np.array(sdp_stats['zs'][1])).reshape(n,n,n,n)
            TZ_list = []
            for (i,v) in enumerate(extra_constraints):
                zsi = mat_real_to_cplx(np.array(sdp_stats['zs'][2+i]))
                TZ_list.append(v['R*'](zsi))
            TZ_sum = np.sum(TZ_list, axis=0)
            T = zs1 - I_ot_rho + TZ_sum

            # some sanity checks to make sure the output makes sense
            verify_tol=1e-7
            if verify_tol:
                err = linalg.eigvalsh((Y-Z_sum).reshape(n**2, n**2) - phi_phi)[0]
                if err < -verify_tol: print "WARNING: phi_phi err =", err

                for (i, (v, Z)) in enumerate(zip(extra_vars, Z_list)):
                    M = v['R'](Z) - v['0']
                    err = linalg.eigvalsh(M)[0]
                    if err < -verify_tol: print "WARNING: R(Z%d) err = %g" % (i, err)

                for (i, v) in enumerate(extra_constraints):
                    M = v['R'](Y) - v['0']
                    err = linalg.eigvalsh(M)[0]
                    if err < -verify_tol: print "WARNING: R(Y) err =", err

                maxeig = linalg.eigvalsh(np.trace(Y-Z_sum, axis1=0, axis2=2))[-1].real
                err = abs(xvec[0] - maxeig)
                if err > verify_tol: print "WARNING: t err =", err

                # make sure it is in S*L(A')
                for mat in self.Sp_basis:
                    dp = np.tensordot(Y, mat.conjugate(), axes=[[0, 2], [0, 1]])
                    err = linalg.norm(dp)
                    if err > 1e-10: print "S err:", err
                    assert err < 1e-10

                # Test the primal solution
                for mat in np.rollaxis(Y_basis, -1):
                    dp = np.tensordot(T, mat.conjugate(), axes=4)
                    err = linalg.norm(dp)
                    if err > verify_tol: print "T in Y_basis.perp() err:", err
                err = abs((T-TZ_sum).trace().trace() + 1 - t)
                if err > verify_tol: print "WARNING: primal vs dual err =", err
                T_plus_Irho = T - TZ_sum + I_ot_rho
                err = linalg.eigvalsh(T_plus_Irho.reshape(n**2, n**2))[0]
                if err < -verify_tol: print "WARNING: T_plus_Irho pos err =", err
                err = abs(np.trace(rho) - 1)
                if err < -verify_tol: print "WARNING: Tr(rho) err =", err
                err = linalg.eigvalsh(rho)[0]
                if err < -verify_tol: print "WARNING: rho pos err =", err

            if long_return:
                ret = {}
                for key in [
                        'n', 'x_to_Y', 'x_to_Z',
                        'phi_phi', 'c', 't', 'Y', 'Z_list', 'xvec', 'sdp_stats',
                        'T', 'rho', 'T_plus_Irho', 'TZ_list', 'TZ_sum',
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
                        'phi_phi', 'c', 't', 'xvec', 'sdp_stats'
                    ]:
                        ret[key] = locals()[key]
                return ret
            else:
                return t
        else:
            raise Exception('cvxopt.sdp returned error: '+sdp_stats['status'])

    def unified_primal(self, T_basis, extra_constraints, extra_vars, long_return=False):
        r"""
        FIXME - only Lovasz for now
        Compute Lovasz/Schrijver/Szegedy type quantities.

        max <\Phi|T-Z + I \ot \rho|\Phi> s.t.
            \rho \succeq 0, \Tr(\rho)=1
            T-Z + I \ot \rho \succeq 0
            T \in S^\perp \ot \mathcal{L}
            R(Z) \in \sum( extra_vars )
            R(T) \in \cap( extra_constraints )
        """

        n = self.n

        Tb_len = T_basis.shape[4]

        # rhotf is the trace-free component of the actual rho
        rhotf_basis = TensorSubspace.from_span([np.eye(n)]).perp(). \
            hermitian_basis().transpose((1,2,0))
        rb_len = rhotf_basis.shape[2]
        assert rb_len == n*n-1

        xvec_len = Tb_len + rb_len + np.sum([ v['basis'].shape[-1] for v in extra_vars])

        idx = 0
        x_to_T = np.zeros((n,n,n,n,xvec_len), dtype=complex)
        x_to_T[:,:,:,:,idx:idx+Tb_len] = T_basis
        idx += Tb_len

        x_to_rhotf = np.zeros((n,n,xvec_len), dtype=complex)
        x_to_rhotf[:,:,idx:idx+rb_len] = rhotf_basis
        idx += rb_len

        x_to_Z = []
        for v in extra_vars:
            xZ = np.zeros((n,n,n,n,xvec_len), dtype=complex)
            Zb_len = v['basis'].shape[-1]
            xZ[:,:,:,:,idx:idx+Zb_len] = v['basis']
            idx += Zb_len
            x_to_Z.append(xZ)

        assert idx == xvec_len

        # T + I \ot rhotf
        x_to_sum = x_to_T + \
                np.tensordot(np.eye(n), x_to_rhotf, axes=0).transpose((0,2,1,3,4))
        for xZ in x_to_Z:
            x_to_sum -= xZ

        # rho \succeq 0
        Fx_1 = x_to_rhotf
        F0_1 = -np.eye(n)/n

        # T + I \ot rho \succeq 0
        Fx_2 = x_to_sum.reshape(n**2, n**2, xvec_len)
        for i in range(Fx_2.shape[2]):
            assert linalg.norm(Fx_2[:,:,i] - Fx_2[:,:,i].conj().T) < 1e-10
        F0_2 = -np.eye(n**2)/n

        c = -np.trace(np.trace(x_to_sum)).real

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
            Fx = np.array([ v['R'](y) for y in np.rollaxis(x_to_T, -1) ], dtype=complex)
            Fx = np.rollaxis(Fx, 0, len(Fx.shape))
            F0 = v['0']
            Fx_econs.append(Fx)
            F0_econs.append(F0)

        Fx_list = [Fx_1, Fx_2] + Fx_evars + Fx_econs
        F0_list = [F0_1, F0_2] + F0_evars + F0_econs

        (xvec, sdp_stats) = call_sdp(c, Fx_list, F0_list)

        if sdp_stats['status'] == 'optimal':
            t = -np.dot(c, xvec) + 1
            T = np.dot(x_to_T, xvec)
            the_sum = np.dot(x_to_sum, xvec)
            rho = np.dot(x_to_rhotf, xvec) + np.eye(n)/n

            # FIXME - construct dual solution
            # FIXME - test primal and dual solutions

            if long_return:
                ret = {}
                for key in [
                        'Fx_list', 'F0_list',
                        'n', 'x_to_T', 'x_to_rhotf', 'the_sum',
                        'c', 't', 'T', 'rho', 'xvec', 'sdp_stats'
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
                        'n', 'x_to_T', 'x_to_rhotf',
                        'c', 't', 'T', 'rho', 'xvec', 'sdp_stats'
                    ]:
                        ret[key] = locals()[key]
                return ret
            else:
                return t
        else:
            raise Exception('cvxopt.sdp returned error: '+sdp_stats['status'])


#if __name__ == "__main__":
#    cvxopt.solvers.options['show_progress'] = False
#    # Unfortunately, Schrijver doesn't converge well.
#    cvxopt.solvers.options['abstol'] = float(1e-5)
#
#    S = NoncommutativeGraph.random(3, 3)
#    print S
#
#    vals = S.get_five_values()
#    for v in vals:
#        print v

# If this module is run from the command line, run the doctests.
if __name__ == "__main__":
    # Doctests require not getting progress messages from SDP solver.
    cvxopt.solvers.options['show_progress'] = False

# FIXME
#    print "Running doctests."
#
#    import doctest
#    doctest.testmod()

    # seed=5 NoncommutativeGraph.random(4, 5) gives gap between Szegedy with positive and with
    # just Hermitian.
    n = 4
    np.random.seed(5)
    S = NoncommutativeGraph.random(n, 12)
    # FIXME - Unfortunately, Schrijver doesn't converge well.
    cvxopt.solvers.options['abstol'] = float(1e-5)
    cvxopt.solvers.options['reltol'] = float(1e-5)
    #cvxopt.solvers.options['show_progress'] = True

    #w = S.szegedy(False)
    #print w
    #v = []
    #v.append(S.cond_psd)
    ##v.append(S.cond_ppt)
    #a = S.unified_primal(S.T_basis_dh, [], v, True)
    #print a['t']
    #print a['t'] - w

    #t_honly = S.unified_dual(S.Y_basis_dh, [], [], False)
    #print 'hermit only:', t_honly

    b = S.szegedy(True, long_return=True)
    #b = S.lovasz_theta(long_return=True)
    locals().update(b)
    print t

    zs1 = mat_real_to_cplx(np.array(sdp_stats['zs'][1])).reshape(n,n,n,n) * 2
    zs2 = mat_real_to_cplx(np.array(sdp_stats['zs'][2])).reshape(n,n,n,n) * 2
    #zs3 = mat_real_to_cplx(np.array(sdp_stats['zs'][3])).reshape(n,n,n,n) * 2
    print (T-TZ_sum).trace().trace()+1 - t
    w=[np.tensordot(x, y.conj(), axes=([],[])).transpose((0, 2, 1, 3)) for x in S.S_basis for y in S.S_basis]
    print np.max([linalg.norm(np.tensordot(T, (x + x.transpose(1,0,3,2).conj()).conj(), axes=4)) for x in w])
