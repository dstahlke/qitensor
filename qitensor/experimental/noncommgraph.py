# Noncommutative graphs as defined by Duan, Severini, Winter in arXiv:1002.2514.

from __future__ import print_function, division

import numpy as np
import scipy.linalg as linalg
import itertools
import cvxopt.base
import cvxopt.solvers

from qitensor import qudit, HilbertSpace, HilbertArray
from qitensor.space import _shape_product
from qitensor.superop import CP_Map
from qitensor.subspace import TensorSubspace

# This is the only thing that is exported.
__all__ = ['NoncommutativeGraph']

# Adapted from
# https://wiki.python.org/moin/PythonDecoratorLibrary#Cached_Properties
class cached_property(object):
    def __init__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__

    def __get__(self, obj, objtype):
        try:
            cache = obj._cache
        except AttributeError:
            cache = obj._cache = {}
        if not self.__name__ in cache:
            cache[self.__name__] = self.fget(obj)
        return cache[self.__name__]

### Some helper functions for cvxopt ###

def mat_cplx_to_real(cmat):
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

    (Fx_list, F0_list) = make_F_real(Fx_list, F0_list)
    Gs = [cvxopt.base.matrix(Fx.reshape(Fx.shape[0]**2, Fx.shape[2])) for Fx in Fx_list]
    hs = [cvxopt.base.matrix(F0) for F0 in F0_list]

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

def tensor_to_matrix(M):
    assert (len(M.shape) % 2 == 0)
    l = len(M.shape) // 2
    nr = _shape_product(M.shape[:l])
    nc = _shape_product(M.shape[l:])
    assert nr == nc
    return M.reshape(nr, nc)

def check_psd(M):
    """
    By how much does M fail to be PSD?
    """

    if isinstance(M, HilbertArray):
        M = M.nparray

    M = tensor_to_matrix(M)

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
        S_flat = self.S_flat = S._op_flatten()
        assert S_flat._col_shp[0] == S_flat._col_shp[1]
        n = self.n = S_flat._col_shp[0]

        assert np.eye(n) in S_flat

        self.cond_psd = {
            'name': 'psd',
            'R':  lambda Z: Z.transpose((0,2,1,3)).reshape(n**2, n**2),
            'R*': lambda Z: Z.reshape(n,n,n,n).transpose((0,2,1,3)),
            '0': np.zeros((n**2, n**2), dtype=complex),
        }

        self.cond_ppt = {
            'name': 'ppt',
            'R':  lambda Z: Z.transpose((1,2,0,3)).reshape(n**2, n**2),
            'R*': lambda Z: Z.reshape(n,n,n,n).transpose((2,0,1,3)),
            '0': np.zeros((n**2, n**2), dtype=complex),
        }

        for c in [self.cond_psd, self.cond_ppt]:
            m = np.random.random((n**2, n**2))
            m2 = c['R'](c['R*'](m))
            assert linalg.norm(m-m2) < 1e-10

    @cached_property
    def S_basis(self):
        ret = np.array(self.S_flat.hermitian_basis()) \
            if self.S_flat.dim() else np.zeros((0, self.n, self.n), dtype=complex)
        assert len(ret.shape) == 3
        return ret

    @cached_property
    def Sp_basis(self):
        ret = np.array(self.S_flat.perp().hermitian_basis()) \
            if self.S_flat.perp().dim() else np.zeros((0, self.n, self.n), dtype=complex)
        assert len(ret.shape) == 3
        return ret

    @cached_property
    def Y_basis(self):
        self.Y_basis = self._get_S_ot_L_basis(self.S_basis)

    @cached_property
    def Y_basis_dh(self):
        return self._basis_doubly_hermit(self.S_basis)

    @cached_property
    def T_basis(self):
        return self._get_S_ot_L_basis(self.Sp_basis)

    @cached_property
    def T_basis_dh(self):
        return self._basis_doubly_hermit(self.Sp_basis)

    @cached_property
    def full_basis_dh(self):
        return self._basis_doubly_hermit(TensorSubspace.full((self.n, self.n)).hermitian_basis())

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

    # FIXME - deprecate
    @classmethod
    def random(cls, spc, num_seeds):
        if isinstance(spc, HilbertSpace):
            n = spc.dim()
        else:
            n = spc
        Sp = TensorSubspace.create_random_hermitian(spc, n**2 - num_seeds - 1, tracefree=True)
        S = Sp.perp()
        assert S.dim() == num_seeds+1
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

        return self.unified_dual(self.Y_basis, [], long_return)

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

        return self.unified_dual(self.Y_basis_dh, v, long_return)

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

        # FIXME
        return [
            self.schrijver(True),
            self.schrijver(False),
            self.lovasz_theta(),
            self.szegedy(False),
            self.szegedy(True),
        ]

    def unified_dual(self, Y_basis, extra_constraints, long_return=False):
        """
        # FIXME - use for Szegedy only now

        min t s.t.
            tI - Tr_A Y \succeq 0
            Y \in S \ot \mathcal{L}
            Y \succeq \Phi
            R(Y) \in \cap( extra_constraints )
        """

        n = self.n

        Yb_len = Y_basis.shape[4]

        # x = [t, Y.A:Si * Y.A':i * Y.A':j]
        xvec_len = 1 + Yb_len

        idx = 1
        x_to_Y = np.zeros((n,n,n,n,xvec_len), dtype=complex)
        x_to_Y[:,:,:,:,idx:idx+Yb_len] = Y_basis
        idx += Yb_len

        assert idx == xvec_len

        phi_phi = np.zeros((n,n, n,n), dtype=complex)
        for (i, j) in itertools.product(list(range(n)), repeat=2):
            phi_phi[i, i, j, j] = 1
        phi_phi = phi_phi.reshape(n**2, n**2)

        # Cost vector.
        # x = [t, Y.A:Si * Y.A':i * Y.A':j]
        c = np.zeros(xvec_len)
        c[0] = 1

        # tI - tr_A(Y) >= 0
        Fx_1 = np.trace(x_to_Y, axis1=0, axis2=2)
        for i in range(n):
            Fx_1[i, i, 0] = -1
        F0_1 = np.zeros((n, n))

        # Y  >=  |phi><phi|
        Fx_2 = -x_to_Y.reshape(n**2, n**2, xvec_len).copy()
        F0_2 = -phi_phi

        Fx_econs = []
        F0_econs = []
        for v in extra_constraints:
            Fx = -np.array([ v['R'](y) for y in np.rollaxis(x_to_Y, -1) ], dtype=complex)
            Fx = np.rollaxis(Fx, 0, len(Fx.shape))
            F0 = -v['0']
            Fx_econs.append(Fx)
            F0_econs.append(F0)

        Fx_list = [Fx_1, Fx_2] + Fx_econs
        F0_list = [F0_1, F0_2] + F0_econs

        (xvec, sdp_stats) = call_sdp(c, Fx_list, F0_list)

        if sdp_stats['status'] in ['optimal', 'primal infeasible']:
            rho = mat_real_to_cplx(np.array(sdp_stats['zs'][0]))
            I_ot_rho = np.tensordot(np.eye(n), rho, axes=0).transpose(0,2,1,3)
            zs1 = mat_real_to_cplx(np.array(sdp_stats['zs'][1])).reshape(n,n,n,n)
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
                if err > verify_tol: print("WARNING: T_plus_Irho pos err =", err)
                err = abs(np.trace(rho) - 1)
                if err > verify_tol: print("WARNING: Tr(rho) err =", err)
                err = check_psd(rho)
                if err > verify_tol: print("WARNING: rho pos err =", err)

                for (i, (v, L)) in enumerate(zip(extra_constraints, L_list)):
                    M = v['R'](L) - v['0']
                    err = check_psd(M)
                    if err > verify_tol: print("WARNING: R(L_%d) err = %g" % (i, err))

        if sdp_stats['status'] == 'optimal':
            t = xvec[0]
            Y = np.dot(x_to_Y, xvec)

            # Verify primal/dual solution
            verify_tol=1e-7
            if verify_tol:
                err = abs(T.trace().trace() + 1 - t)
                if err > verify_tol: print("WARNING: primal vs dual err =", err)

                err = check_psd(Y.reshape(n**2, n**2) - phi_phi)
                if err > verify_tol: print("WARNING: phi_phi err =", err)

                for (i, v) in enumerate(extra_constraints):
                    M = v['R'](Y) - v['0']
                    err = check_psd(M)
                    if err > verify_tol: print("WARNING: R(Y) [%d] err = %g" %(i, err))

                maxeig = linalg.eigvalsh(np.trace(Y, axis1=0, axis2=2))[-1].real
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
                    'n', 'x_to_Y',
                    'phi_phi', 'c', 't', 'Y', 'xvec', 'sdp_stats',
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

    def _get_cone_set(self, cones):
        if isinstance(cones, list):
            return cones
        else:
            assert isinstance(cones, str)
            return {
                'hermit': [],
                'psd': [self.cond_psd],
                'psd&ppt': [self.cond_psd, self.cond_ppt],
                'ppt': [self.cond_ppt],
            }[cones]

    def schrijver(self, cones, long_return=False):
        r"""
        My non-commutative generalization of Schrijver's number.

        max <\Phi|T + I \ot \rho|\Phi> s.t.
            \rho \succeq 0, \Tr(\rho)=1
            T + I \ot \rho \succeq 0
            T \in S^\perp \ot \bar{S}^\perp
            T^\ddag = T
            R(T) \in cones

        min ||Tr_A(Y)|| s.t.
            Y \succeq |\Phi><\Phi|
            Y+(L+L^\dag)-X \in S \djp \bar{S}
            R(L) \in cones^*
            X^\ddag = -X
            X^\dag = X

        If the long_return option is True, then some extra status and internal
        quantities are returned (such as the optimal Y operator).
        """

        # FIXME - test with S being full-space

        cones = self._get_cone_set(cones)

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
        Fx_1 = -x_to_rhotf
        F0_1 = np.eye(n)/n

        # T + I \ot rho \succeq 0
        Fx_2 = -x_to_sum.reshape(n**2, n**2, xvec_len)
        for i in range(Fx_2.shape[2]):
            assert linalg.norm(Fx_2[:,:,i] - Fx_2[:,:,i].conj().T) < 1e-10
        F0_2 = np.eye(n**2)/n

        c = -np.trace(np.trace(x_to_sum)).real

        Fx_econs = []
        F0_econs = []
        for v in cones:
            Fx = -np.array([ v['R'](y) for y in np.rollaxis(x_to_T, -1) ], dtype=complex)
            Fx = np.rollaxis(Fx, 0, len(Fx.shape))
            F0 = -v['0']
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

            Y = J + mat_real_to_cplx(np.array(sdp_stats['zs'][1])).reshape(n,n,n,n)

            L_list = []
            zs_idx = 2
            for (i,v) in enumerate(cones):
                zsi = mat_real_to_cplx(np.array(sdp_stats['zs'][2+i]))
                zs_idx += 1
                # over 2 because we will later do L+L^\dag
                L = v['R*'](zsi) / 2
                L_list.append(L)
            assert zs_idx == len(sdp_stats['zs'])

            # cvxopt guarantees the dual to give zero here
            #zs0 = mat_real_to_cplx(np.array(sdp_stats['zs'][0]))
            #foo = (
            #    np.tensordot(zs0, x_to_rhotf.conj(), axes=2) +
            #    np.tensordot(Y, x_to_sum.conj(), axes=4)
            #)
            #if len(L_list):
            #    foo += np.tensordot(np.sum(L_list, axis=0), x_to_T.conj(), axes=4)
            #print('**', linalg.norm(foo))

            # Copy rot-antihermit portion of Y to X.
            X = Y - project_dh(Y)

            verify_tol=1e-7
            if verify_tol:
                err = {}

                # Test the primal solution
                err[r'T + I \ot rho'] = linalg.norm(T + I_rho - T_plus_Irho)
                err['primal value'] = abs(t - T_plus_Irho.trace(axis1=0, axis2=1).trace(axis1=0, axis2=1))

                for mat in self.S_basis:
                    dp = np.tensordot(T, mat.conj(), axes=[[0, 2], [0, 1]])
                    err[r'T in S^\perp \ot \bar{S}^\perp'] = linalg.norm(dp)

                err['T_plus_Irho PSD'] = check_psd(T_plus_Irho.reshape(n**2, n**2))
                err['Tr(rho)'] = abs(np.trace(rho) - 1)
                err['rho PSD'] = check_psd(rho)

                for (i, v) in enumerate(cones):
                    M = v['R'](T) - v['0']
                    err['R(T) in '+v['name']] = check_psd(M)

                # Test the dual solution
                err['dual value'] = abs(t - linalg.eigvalsh(Y.trace(axis1=0, axis2=2))[-1])

                err_Y_space = 0
                for matA in self.Sp_basis:
                    for matB in self.Sp_basis:
                        xy = np.tensordot(matA, matB.conj(), axes=([],[])).transpose((0, 2, 1, 3))
                        YLX = Y-X
                        if len(L_list):
                            L_sum = np.sum(L_list, axis=0)
                            L_sum = L_sum + L_sum.transpose(2,3,0,1).conj()
                            YLX += L_sum
                        dp = np.tensordot(YLX, xy.conj(), axes=4)
                        err_Y_space += abs(dp)
                err[r'Y+L-X in S \djp \bar{S}'] = err_Y_space

                for (i, (v, L)) in enumerate(zip(cones, L_list)):
                    M = v['R'](L) - v['0']
                    err['R(L) in '+v['name']] = check_psd(M)

                err['Y-J PSD'] = check_psd((Y-J).reshape(n*n, n*n))

                err['X - X.H'] = linalg.norm(X - X.transpose(2,3,0,1).conj())
                err['X + X^ddag'] = linalg.norm(X + X.transpose(1,0,3,2).conj())

                assert min(err.values()) >= 0
                for (k, v) in err.items():
                    if v > verify_tol:
                        print('WARRNING: err[%s] = %g' % (k, v))

            if long_return:
                L_map = { C['name']: L for (C, L) in zip(cones, L_list) }
                if self.S._hilb_space is not None:
                    ha = self.S._hilb_space.ket_space()
                    hb = ha.prime # FIXME - what if ha=|a,a'>
                    rho = hb.O.array(rho, reshape=True)
                    T = (ha*hb).O.array(T, reshape=True)
                    Y = (ha*hb).O.array(Y, reshape=True)
                    L_map = { k: (ha*hb).O.array(L_map[k], reshape=True) for k in L_map.keys() }
                    X = (ha*hb).O.array(X, reshape=True)
                else:
                    ha = None
                    hb = None
                #return locals()
                ret = {}
                for key in [
                    't', 'T', 'rho', 'Y', 'L_map', 'X', 'ha', 'hb', 'sdp_stats'
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

def test_schrijver(dA=3, dS=5, seed=1):
    """
    >>> test_schrijver(3, 5, 1)
    --- Schrijver with hermit
    t = 3.31460511605
    total err: 6.40497175329e-16
    total err: 3.95143932875e-15
    duality gap: 3.51548878896e-09
    --- Schrijver with psd
    t = 3.31460511899
    total err: 9.52374539775e-10
    total err: 2.0912092081e-09
    duality gap: 2.17725570906e-09
    --- Schrijver with ppt
    t = 2.42819820943
    total err: 1.11928504243e-10
    total err: 2.07485612315e-10
    duality gap: -1.67085012492e-10
    --- Schrijver with psd&ppt
    t = 2.42819821122
    total err: 1.90901022447e-10
    total err: 5.81738926097e-10
    duality gap: -1.90005344791e-10
    """

    ha = qudit('a', dA)
    np.random.seed(seed)
    S = TensorSubspace.create_random_hermitian(ha, dS, tracefree=True).perp()
    G = NoncommutativeGraph(S)

    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = float(1e-8)
    cvxopt.solvers.options['reltol'] = float(1e-8)

    # FIXME - are the first two and last two always the same?
    for cone in ('hermit', 'psd', 'ppt', 'psd&ppt'):
        print('--- Schrijver with', cone)
        info = G.schrijver(cone, True)
        print('t =', info['t'])
        (tp, errp) = check_schrijver_primal(S, cone, *[ info[x] for x in 'ha,hb,rho,T'.split(',') ], report=True)
        (td, errd) = check_schrijver_dual(S, cone, *[ info[x] for x in 'ha,hb,Y,L_map,X'.split(',') ], report=True)
        print('duality gap:', td-tp)

    return info

def check_schrijver_primal(S, cones, ha, hb, rho, T, report=False):
    r"""
    Verify Schrijver primal solution.
    Returns ``(t, err)`` where ``t`` is the value and ``err`` is the amount by which
    feasibility constrains are violated.
    """

    return _checking_routine(S, cones, ha, hb, { 'schrijver_primal': (rho, T) }, report)

def check_schrijver_dual(S, cones, ha, hb, Y, L_map, X, report=False):
    r"""
    Verify Schrijver dual solution.
    Returns ``(t, err)`` where ``t`` is the value and ``err`` is the amount by which
    feasibility constrains are violated.
    """

    return _checking_routine(S, cones, ha, hb, { 'schrijver_dual': (Y, L_map, X) }, report)

def _checking_routine(S, cones, ha, hb, task, report):
    assert S._hilb_space == ha.O
    ncg = NoncommutativeGraph(S)

    cones = ncg._get_cone_set(cones)
    cone_names = frozenset(C['name'] for C in cones)

    def R(x):
        return x.relabel({ hb: ha.H, ha.H: hb })

    def ddag(x):
        ret = x.relabel({ ha: hb, hb: ha, ha.H: hb.H, hb.H: ha.H }).conj()
        assert (ret - R(R(x).H)).norm() < 1e-12
        return ret

    Sb = S.map(lambda x: x.relabel({ ha: hb, ha.H: hb.H }).conj())

    Phi = ha.eye().relabel({ ha.H: hb })
    J = Phi.O

    def proj_Sp_ot_Sp(x):
        ret = (ha*hb).O.array()
        for pa in S.perp():
            foo = (x * pa.H).trace(ha)
            ret += pa * Sb.perp().project(foo)
        return ret

    err = {}

    if 'schrijver_primal' in task:
        (rho, T) = task['schrijver_primal']

        assert rho.space == hb.O
        assert T.space == (ha*hb).O
        val = 1 + (Phi.H * T * Phi).real
        err[r'trace(rho)'] = abs(1 - rho.trace())
        err[r'rho PSD'] = check_psd(rho)
        err[r'T + I \ot rho PSD'] = check_psd(T + ha.eye()*rho)
        err[r'T^\ddag - T'] = (T - ddag(T)).norm()

        for C in cone_names:
            if C == 'psd':
                err[r'T_PSD'] = check_psd(R(T))
            elif C == 'ppt':
                err[r'T_PPT'] = check_psd(R(T).transpose(ha))
            else:
                assert 0

        # Slow
        #Sp_ot_Sp = S.perp() * Sb.perp()
        #err[r'T \in S^\perp \ot \bar{S}^\perp'] = linalg.norm(Sp_ot_Sp.perp().to_basis(T))
        # Faster
        err[r'T \in S^\perp \ot \bar{S}^\perp'] = (T - proj_Sp_ot_Sp(T)).norm()

    if 'schrijver_dual' in task:
        (Y, L_map, X) = task['schrijver_dual']

        val = Y.trace(ha).eigvalsh()[-1]

        err[r'Y \succeq J'] = check_psd(Y - J)

        YLX = Y - X
        if len(cone_names):
            L_sum = np.sum(list(L_map.values()))
            YLX += L_sum + L_sum.H

        # Slow
        #S_djp_S = S * hb.O.full_space() | ha.O.full_space() * Sb
        #err[r'Y+L-X \in S \djp \bar{S}'] = linalg.norm(S_djp_S.perp().to_basis(YLX))
        # Faster
        err[r'T \in S^\perp \ot \bar{S}^\perp'] = proj_Sp_ot_Sp(YLX).norm()

        for C in cone_names:
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

    if report:
        for (k, v) in err.items():
            if v > 1e-7:
                print('err[%s] = %g' % (k, v))

        print('total err:', sum(err.values()))

    return (val, err)

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
    #A1 = (X + X.H) / 2
    #A2 = X - A1
    #A11 = (A1 + ddag(A1)) / 2
    #A12 = A1 - A11
    #A21 = (A2 + ddag(A2)) / 2
    #A22 = A2 - A21
    #print(A11.norm())
    #print(A12.norm())
    #print(A21.norm())
    #print(A22.norm())

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

    #t_honly = S.unified_dual(S.Y_basis_dh, [], False)
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
