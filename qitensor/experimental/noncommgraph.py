# Noncommutative graphs as defined by Duan, Severini, Winter in arXiv:1002.2514.

from __future__ import print_function, division

import numpy as np
import scipy.linalg as linalg
import itertools
import collections
import cvxopt.base
import cvxopt.solvers

from qitensor import qudit, HilbertSpace, HilbertArray
from qitensor.space import _shape_product
from qitensor.superop import CP_Map
from qitensor.subspace import TensorSubspace

__all__ = [
    'from_adjmat',
    'from_graph6',
    'pentagon',
    'make_channel',
    'from_channel',
    'lovasz_theta',
    'szegedy',
    'schrijver',
    'qthperp',
    'qthmperp',
    'qthpperp',
    'get_many_values',
]

# Adapted from
# https://wiki.python.org/moin/PythonDecoratorLibrary#Cached_Properties
class cached_property(object):
    def __init__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__

    def __get__(self, obj, objtype):
        # For Sphinx:
        if obj is None:
            return self.fget

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

### Helper class, for private use ######################

class GraphProperties(object):
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
        }

        self.cond_ppt = {
            'name': 'ppt',
            'R':  lambda Z: Z.transpose((1,2,0,3)).reshape(n**2, n**2),
            'R*': lambda Z: Z.reshape(n,n,n,n).transpose((2,0,1,3)),
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
        return self.get_S_ot_L_basis(self.S_basis)

    @cached_property
    def Y_basis_dh(self):
        return self.basis_doubly_hermit(self.S_basis)

    @cached_property
    def T_basis(self):
        return self.get_S_ot_L_basis(self.Sp_basis)

    @cached_property
    def T_basis_dh(self):
        return self.basis_doubly_hermit(self.Sp_basis)

    @cached_property
    def full_basis_dh(self):
        return self.basis_doubly_hermit(TensorSubspace.full((self.n, self.n)).hermitian_basis())

    @cached_property
    def top_space(self):
        return self.S._hilb_space.ket_space()

    @cached_property
    def bottom_space(self):
        ret = self.top_space
        while not ret.isdisjoint(self.top_space):
            ret = ret.prime
        return ret

    def R(self, x):
        ha = self.top_space
        hb = self.bottom_space
        return x.relabel({ hb: ha.H, ha.H: hb })

    def ddag(self, x):
        ha = self.top_space
        hb = self.bottom_space
        ret = x.relabel({ ha: hb, hb: ha, ha.H: hb.H, hb.H: ha.H }).conj()
        assert (ret - self.R(self.R(x).H)).norm() < 1e-12
        return ret

    def make_ab_array(self, M):
        ha = self.top_space
        hb = self.bottom_space
        return (ha*hb).O.array(M, reshape=True, input_axes=ha.axes+hb.axes+ha.H.axes+hb.H.axes)

    def __str__(self):
        return '<GraphProperties of '+self.S._str_inner()+'>'

    def __repr__(self):
        return str(self)

    def get_S_ot_L_basis(self, Sb):
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

    def basis_doubly_hermit(self, Sb):
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

    def test_get_Y_basis_doubly_hermit(self):
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

    def test_doubly_hermitian_basis(self):
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

    def get_cone_set(self, cones):
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

### End of GraphProperties #################################

def from_adjmat(adj_mat, loops=False):
    """
    Create a non-commutative graph from the adjacency matrix of a classical graph.

    The given adjacency matrix must be symmetric.

    >>> import numpy
    >>> # 5-cycle graph
    >>> adj_mat = np.array([
    ...     [0, 1, 0, 0, 1],
    ...     [1, 0, 1, 0, 0],
    ...     [0, 1, 0, 1, 0],
    ...     [0, 0, 1, 0, 1],
    ...     [1, 0, 0, 1, 0]
    ... ])
    >>> S = from_adjmat(adj_mat)
    >>> S
    <TensorSubspace of dim 10 over space (5, 5)>
    >>> theta = lovasz_theta(~S)
    >>> abs(theta - numpy.sqrt(5)) < 1e-8
    True
    >>> T = from_adjmat(adj_mat, loops=True)
    >>> T
    <TensorSubspace of dim 15 over space (5, 5)>
    >>> theta = lovasz_theta(T)
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

    ret = TensorSubspace.from_span(basis)

    if loops:
        ret |= TensorSubspace.diagonals((n, n))

    return ret

def from_graph6(g6, loops=False):
    """
    Create a non-commutative graph corresponding to the classical graph with the given
    graph6 code.

    >>> from_graph6('GRddY{')
    <TensorSubspace of dim 32 over space (8, 8)>
    >>> from_graph6('GRddY{', loops=True)
    <TensorSubspace of dim 40 over space (8, 8)>
    """

    import networkx as nx
    G = nx.parse_graph6(g6)
    n = len(G)
    basis = []

    for (i, j) in G.edges():
        m = np.zeros((n, n), dtype=complex)
        m[i, j] = 1
        basis.append(m)
    basis += [ x.transpose() for x in basis ]

    ret = TensorSubspace.from_span(basis)

    if loops:
        ret |= TensorSubspace.diagonals((n, n))

    return ret

def pentagon():
    """
    Create the 5-cycle graph.  Useful for testing.

    >>> import numpy
    >>> # 5-cycle graph
    >>> adj_mat = np.array([
    ...     [0, 1, 0, 0, 1],
    ...     [1, 0, 1, 0, 0],
    ...     [0, 1, 0, 1, 0],
    ...     [0, 0, 1, 0, 1],
    ...     [1, 0, 0, 1, 0]
    ... ])
    >>> S = from_adjmat(adj_mat)
    >>> T = pentagon()
    >>> S.equiv(T)
    True
    """

    # Adjacency matric for the 5-cycle graph.
    adj_mat = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])
    return from_adjmat(adj_mat)

def make_channel(S):
    """
    Makes a CPTP map whose confusibility graph is equal to ``S``.

    >>> ha = qudit('a', 3)
    >>> S = TensorSubspace.create_random_hermitian(ha, 5, tracefree=True).perp()
    >>> S
    <TensorSubspace of dim 4 over space (|a><a|)>
    >>> N = make_channel(S)
    >>> N
    CP_Map( |a><a| to |a,k><a,k| )
    >>> T = from_channel(N)
    >>> T
    <TensorSubspace of dim 4 over space (|a><a|)>
    >>> T.equiv(S)
    True
    """

    assert S._hilb_space is not None
    spc = S._hilb_space
    assert S.is_hermitian()
    assert spc.eye() in S

    if S.dim() == 1:
        return CP_Map.identity(spc)

    # Hermitian basis for all but identity
    B = (S - spc.eye().span()).hermitian_basis()
    # Add identity to make a PSD basis
    B = [ b - spc.eye() * b.eigvalsh()[0] for b in B ]
    # Make everything in the basis sum to a bounded operator
    m = np.sum(B).eigvalsh()[-1]
    B = [ b/m for b in B ]
    # Add a final element to make a complete POVM
    B += [ spc.eye() - np.sum(B) ]
    # Make Kraus operators
    J = [ b.sqrt() for b in B ]
    # Remove a unitary degree of freedom
    # Is this needed or just here to make a nicer answer?
    J = [ j.svd()[0].H * j for j in J ]

    # Add basis index to channel output (eliminates interference between different basis
    # elements).
    hk = qudit('k', len(J))
    J = [ hk.ket(k) * j for (k,j) in enumerate(J) ]

    Kspc = TensorSubspace.from_span(J)
    assert S.equiv(Kspc.H * Kspc)

    chan = CP_Map.from_kraus(J)
    assert chan.is_cptp()

    return chan

def from_channel(N):
    """
    Returns the confusibility graph for a ``CP_Map``.

    >>> ha = qudit('a', 3)
    >>> N = CP_Map.decohere(ha)
    >>> N
    CP_Map( |a><a| to |a><a| )
    >>> S = from_channel(N)
    >>> S
    <TensorSubspace of dim 3 over space (|a><a|)>
    >>> theta = lovasz_theta(S)
    >>> abs(theta - 3.0) < 1e-8
    True
    """

    assert N.is_cptp()
    K = TensorSubspace.from_span(N.krauses())
    return K.H * K

### Main SDP routines =============

def lovasz_theta(S, long_return=False, verify_tol=1e-7):
    """
    Compute the non-commutative generalization of the Lovasz function,
    using Theorem 9 of arXiv:1002.2514.

    If the long_return option is True, then the optimal solution (T, Y, etc.) is returned.
    """

    if S.perp().dim() == 0:
        return 1.0

    ncg = GraphProperties(S)
    n = ncg.n

    Yb_len = ncg.Y_basis.shape[4]

    # x = [t, Y.A:Si * Y.A':i * Y.A':j]
    xvec_len = 1 + Yb_len

    idx = 1
    x_to_Y = np.zeros((n,n,n,n,xvec_len), dtype=complex)
    x_to_Y[:,:,:,:,idx:idx+Yb_len] = ncg.Y_basis
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

    # Y  >=  |Phi><Phi|
    Fx_2 = -x_to_Y.reshape(n**2, n**2, xvec_len).copy()
    F0_2 = -phi_phi

    (xvec, sdp_stats) = call_sdp(c, [Fx_1, Fx_2], [F0_1, F0_2])

    if sdp_stats['status'] == 'optimal':
        rho = mat_real_to_cplx(np.array(sdp_stats['zs'][0]))
        I_ot_rho = np.tensordot(np.eye(n), rho, axes=0).transpose(0,2,1,3)
        zs1 = mat_real_to_cplx(np.array(sdp_stats['zs'][1])).reshape(n,n,n,n)
        T = zs1 - I_ot_rho

        t = xvec[0]
        Y = np.dot(x_to_Y, xvec)

        # Verify primal/dual solution
        if verify_tol:
            err = collections.defaultdict(float)

            # Test the primal solution
            for mat in np.rollaxis(ncg.Y_basis, -1):
                dp = np.tensordot(T, mat.conj(), axes=4)
                err[r'T in Ybas.perp()'] += linalg.norm(dp)
            T_plus_Irho = T + I_ot_rho
            err[r'T_plus_Irho pos'] = check_psd(T_plus_Irho.reshape(n**2, n**2))
            err[r'Tr(rho)'] = abs(np.trace(rho) - 1)
            err[r'rho pos'] = check_psd(rho)

            err[r'primal value'] = abs(T.trace().trace() + 1 - t)

            err[r'Y - phi_phi PSD'] = check_psd(Y.reshape(n**2, n**2) - phi_phi)

            maxeig = linalg.eigvalsh(np.trace(Y, axis1=0, axis2=2))[-1].real
            err[r'dual value'] = abs(t - maxeig)

            for mat in ncg.Sp_basis:
                dp = np.tensordot(Y, mat.conj(), axes=[[0, 2], [0, 1]])
                err[r'Y in S \ot \bar{S}'] += linalg.norm(dp)

            assert min(err.values()) >= 0
            for (k, v) in err.items():
                if v > verify_tol:
                    print('WARNING: err[%s] = %g' % (k, v))

        if long_return:
            if ncg.S._hilb_space is not None:
                ha = ncg.top_space
                hb = ncg.bottom_space
                rho = hb.O.array(rho, reshape=True)
                T = ncg.make_ab_array(T)
                Y = ncg.make_ab_array(Y)
            else:
                ha = None
                hb = None
            to_ret = [ 't', 'T', 'rho', 'Y', 'ha', 'hb', 'sdp_stats' ]
            _locals = locals()
            return { key: _locals[key] for key in to_ret }
        else:
            return t
    else:
        raise Exception('cvxopt.sdp returned error: '+sdp_stats['status'])

def szegedy(S, cones, long_return=False, verify_tol=1e-7):
    r"""
    My non-commutative generalization of Szegedy's number.

    .. math::
        \max &\left<\Phi|T + I \otimes \rho|\Phi\right> \textrm{ s.t.} \\
            &\rho \succeq 0, \Tr(\rho)=1 \\
            &T + I \otimes \rho \succeq 0 \\
            &T+\sum_i (L_i+L_i^\dag) \in S^\perp * \overline{S}^\perp \\
            &R(L_i)+R(L_i)^\dag \in \mathcal{C}_i^* \quad \forall i \\
        \min &\opnorm{Tr_A(Y)} \textrm{ s.t.} \\
            &Y \succeq \ket{\Phi}\bra{\Phi} \\
            &Y \in S \otimes \overline{S} \\
            &R(Y) \in \mathcal{C}_i \quad \forall i \\
            &Y = Y^\ddag \textrm{ (redundant with above)}

    ``cones`` can be ``hermit``, ``psd``, ``ppt``, or ``psd&ppt``.

    If the long_return option is True, then the optimal solution (T, Y, etc.) is returned.
    It is possible for the value to be ``inf``, in which case the primal solution will be
    a certificate of infeasibility, with :math:`\Tr(\rho)=0`.
    """

    if S.perp().dim() == 0:
        return 1.0

    ncg = GraphProperties(S)
    cones = ncg.get_cone_set(cones)

    for C in cones:
        assert 'R' in C

    n = ncg.n

    Ybas = ncg.Y_basis_dh
    Yb_len = Ybas.shape[4]

    # x = [t, Y.A:Si * Y.A':i * Y.A':j]
    xvec_len = 1 + Yb_len

    idx = 1
    x_to_Y = np.zeros((n,n,n,n,xvec_len), dtype=complex)
    x_to_Y[:,:,:,:,idx:idx+Yb_len] = Ybas
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

    # Y  >=  |Phi><Phi|
    Fx_2 = -x_to_Y.reshape(n**2, n**2, xvec_len).copy()
    F0_2 = -phi_phi

    Fx_econs = []
    F0_econs = []
    for v in cones:
        Fx = -np.array([ v['R'](y) for y in np.rollaxis(x_to_Y, -1) ], dtype=complex)
        Fx = np.rollaxis(Fx, 0, len(Fx.shape))
        F0 = -np.zeros((n**2, n**2), dtype=complex)
        Fx_econs.append(Fx)
        F0_econs.append(F0)

    Fx_list = [Fx_1, Fx_2] + Fx_econs
    F0_list = [F0_1, F0_2] + F0_econs

    (xvec, sdp_stats) = call_sdp(c, Fx_list, F0_list)

    err = collections.defaultdict(float)

    if sdp_stats['status'] in ['optimal', 'primal infeasible']:
        rho = mat_real_to_cplx(np.array(sdp_stats['zs'][0]))
        I_ot_rho = np.tensordot(np.eye(n), rho, axes=0).transpose(0,2,1,3)
        zs1 = mat_real_to_cplx(np.array(sdp_stats['zs'][1])).reshape(n,n,n,n)
        L_list = []
        for (i,v) in enumerate(cones):
            zsi = mat_real_to_cplx(np.array(sdp_stats['zs'][2+i]))
            # over 2 because we will later do L_i+L_i^\dag
            L_i = v['R*'](zsi) / 2
            L_list.append(L_i)
        T = zs1 - I_ot_rho

        # Copy rot-antihermit portion of T to X.
        X = T - project_dh(T)
        if len(cones):
            L_list[0] -= X/2
        else:
            L_list.append(-X/2)
        X = None

        if sdp_stats['status'] == 'primal infeasible':
            # Rescale certificate of infeasibility so that <Phi|T|Phi>=1 exactly.
            s = T.trace().trace()
            T /= s
            rho /= s
            L_list = [ L_i/s for L_i in L_list ]

        # Verify dual solution (or part of it; more is done below)
        if verify_tol:
            # Test the primal solution
            L = np.sum(L_list, axis=0)
            LH = L.transpose(2,3,0,1).conj()
            TLL = T+L+LH

            for mat in np.rollaxis(Ybas, -1):
                dp = np.tensordot(TLL, mat.conj(), axes=4)
                err[r'T+L+L^\dag in Ybas.perp()'] += linalg.norm(dp)

            if len(cones):
                for (i, (v, L_i)) in enumerate(zip(cones, L_list)):
                    M = v['R'](L_i).copy()
                    M += M.T.conj()
                    err['R(L_i)+R(L_i)^\dag in '+v['name']] = check_psd(M)

            T_plus_Irho = T + I_ot_rho
            err[r'T_plus_Irho pos'] = check_psd(T_plus_Irho.reshape(n**2, n**2))
            if sdp_stats['status'] == 'optimal':
                err[r'Tr(rho)'] = abs(np.trace(rho) - 1)
            else:
                err[r'Tr(rho)'] = abs(np.trace(rho))
            err[r'rho pos'] = check_psd(rho)

            # not mandatory, but we can get this condtion anyway
            TLLddag = TLL.transpose((1,0,3,2)).conj()
            err[r'R(T+L+L^\dag) \in Herm'] = linalg.norm(TLL-TLLddag)

            TH = T.transpose((2,3,0,1)).conj()
            err[r'T-T^\dag'] = linalg.norm(T-TH)

    if sdp_stats['status'] == 'optimal':
        t = xvec[0]
        Y = np.dot(x_to_Y, xvec)

        # Verify primal/dual solution
        if verify_tol:
            err[r'primal value'] = abs(T.trace().trace() + 1 - t)

            err[r'Y - phi_phi PSD'] = check_psd(Y.reshape(n**2, n**2) - phi_phi)

            for v in cones:
                M = v['R'](Y)
                err[r'R(Y) in '+v['name']] = check_psd(M)

            Yddag = Y.transpose((1,0,3,2)).conj()
            err[r'Y-Y^\ddag'] = linalg.norm(Y-Yddag)

            maxeig = linalg.eigvalsh(np.trace(Y, axis1=0, axis2=2))[-1].real
            err[r'dual value'] = abs(t - maxeig)

    if err:
        assert min(err.values()) >= 0
    for (k, v) in err.items():
        if v > verify_tol:
            print('WARNING: err[%s] = %g' % (k, v))

    if sdp_stats['status'] in ['optimal', 'primal infeasible']:
        if sdp_stats['status'] == 'primal infeasible':
            t = np.inf
        if long_return:
            if len(cones):
                L_map = { C['name']: L_i for (C, L_i) in zip(cones, L_list) }
            else:
                assert len(L_list)==1
                L_map = { 'hermit': L_list[0] }
            if ncg.S._hilb_space is not None:
                ha = ncg.top_space
                hb = ncg.bottom_space
                rho = hb.O.array(rho, reshape=True)
                T = ncg.make_ab_array(T)
                if sdp_stats['status'] == 'optimal':
                    Y = ncg.make_ab_array(Y)
                L_map = { k: ncg.make_ab_array(L_map[k]) for k in L_map.keys() }
            else:
                ha = None
                hb = None
            to_ret = [ 't', 'T', 'rho', 'L_map', 'ha', 'hb', 'sdp_stats' ]
            if sdp_stats['status'] == 'optimal':
                to_ret += 'Y'
            _locals = locals()
            return { key: _locals[key] for key in to_ret }
        else:
            return t
    else:
        raise Exception('cvxopt.sdp returned error: '+sdp_stats['status'])

def schrijver(S, cones, long_return=False, verify_tol=1e-7):
    r"""
    My non-commutative generalization of Schrijver's number.

    .. math::
        \max &\left<\Phi|T + I \otimes \rho|\Phi\right> \textrm{ s.t.} \\
            &\rho \succeq 0, \Tr(\rho)=1 \\
            &T + I \otimes \rho \succeq 0 \\
            &T \in S^\perp \otimes \overline{S}^\perp \\
            &R(T) \in \mathcal{C}_i \quad \forall i \\
            &T = T^\ddag \textrm{ (redundant with above)} \\
        \min &\opnorm{Tr_A(Y)} \textrm{ s.t.} \\
            &Y \succeq \ket{\Phi}\bra{\Phi} \\
            &Y+\sum_i (L_i+L_i^\dag) \in S * \overline{S} \\
            &R(L_i)+R(L_i)^\dag \in \mathcal{C}_i^* \quad \forall i

    ``cones`` can be ``hermit``, ``psd``, ``ppt``, or ``psd&ppt``.

    If the long_return option is True, then the optimal solution (T, Y, etc.) is returned.
    """

    if S.perp().dim() == 0:
        return 1.0

    ncg = GraphProperties(S)
    cones = ncg.get_cone_set(cones)

    for C in cones:
        assert 'R' in C

    n = ncg.n

    Tbas = ncg.T_basis_dh
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
        F0 = -np.zeros((n**2, n**2), dtype=complex)
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

        # Copy rot-antihermit portion of Y to X.
        X = Y - project_dh(Y)
        if len(cones):
            L_list[0] -= X/2
        else:
            L_list.append(-X/2)
        X = None

        if verify_tol:
            err = collections.defaultdict(float)

            # Test the primal solution
            err[r'T + I \ot rho'] = linalg.norm(T + I_rho - T_plus_Irho)
            err['primal value'] = abs(t - T_plus_Irho.trace(axis1=0, axis2=1).trace(axis1=0, axis2=1))

            err['T_plus_Irho PSD'] = check_psd(T_plus_Irho.reshape(n**2, n**2))
            err['Tr(rho)'] = abs(np.trace(rho) - 1)
            err['rho PSD'] = check_psd(rho)

            for v in cones:
                M = v['R'](T)
                err['R(T) in '+v['name']] = check_psd(M)

            Tddag = T.transpose((1,0,3,2)).conj()
            err[r'T-T^\ddag'] = linalg.norm(T-Tddag)

            # Test the dual solution
            err['dual value'] = abs(t - linalg.eigvalsh(Y.trace(axis1=0, axis2=2))[-1])

            L = np.sum(L_list, axis=0)
            LH = L.transpose(2,3,0,1).conj()
            YLL = Y+L+LH

            err_Y_space = 0
            for matA in ncg.Sp_basis:
                for matB in ncg.Sp_basis:
                    xy = np.tensordot(matA, matB.conj(), axes=([],[])).transpose((0, 2, 1, 3))
                    dp = np.tensordot(YLL, xy.conj(), axes=4)
                    err_Y_space += abs(dp)
            err[r'Y+L+L^\dag in S \djp \bar{S}'] = err_Y_space

            if len(cones):
                for (i, (v, L_i)) in enumerate(zip(cones, L_list)):
                    M = v['R'](L_i).copy()
                    M += M.T.conj()
                    err['R(L_i) in '+v['name']] = check_psd(M)

            # not mandatory, but we can get this condtion anyway
            YLLddag = YLL.transpose((1,0,3,2)).conj()
            err[r'R(Y+L+L^\dag) \in Herm'] = linalg.norm(YLL-YLLddag)

            YH = Y.transpose((2,3,0,1)).conj()
            err[r'Y-Y^\dag'] = linalg.norm(Y-YH)

            err['Y-J PSD'] = check_psd((Y-J).reshape(n*n, n*n))

            assert min(err.values()) >= 0
            for (k, v) in err.items():
                if v > verify_tol:
                    print('WARNING: err[%s] = %g' % (k, v))

        if long_return:
            if len(cones):
                L_map = { C['name']: L_i for (C, L_i) in zip(cones, L_list) }
            else:
                assert len(L_list)==1
                L_map = { 'hermit': L_list[0] }
            if ncg.S._hilb_space is not None:
                ha = ncg.top_space
                hb = ncg.bottom_space
                rho = hb.O.array(rho, reshape=True)
                T = ncg.make_ab_array(T)
                Y = ncg.make_ab_array(Y)
                L_map = { k: ncg.make_ab_array(L_map[k]) for k in L_map.keys() }
            else:
                ha = None
                hb = None
            to_ret = [ 't', 'T', 'rho', 'Y', 'L_map', 'ha', 'hb', 'sdp_stats' ]
            _locals = locals()
            return { key: _locals[key] for key in to_ret }
        else:
            return t
    else:
        raise Exception('cvxopt.sdp returned error: '+sdp_stats['status'])

### Convenience functions ================

def qthperp(S, long_return=False):
    '''
    Returns ``lovasz_theta(S.perp())``.
    ``S`` should be trace-free.
    '''

    return lovasz_theta(~S, long_return)

def qthmperp(S, cones, long_return=False):
    '''
    Returns ``schrijver(S.perp())``.
    ``S`` should be trace-free.
    '''

    return schrijver(~S, cones, long_return)

def qthpperp(S, cones, long_return=False):
    '''
    Returns ``szegedy(S.perp())``.
    ``S`` should be trace-free.
    '''

    return szegedy(~S, cones, long_return)

def get_many_values(S, which_ones=None):
    """
    >>> np.random.seed(1)
    >>> S = TensorSubspace.create_random_hermitian(3, 5, tracefree=True).perp()
    >>> cvxopt.solvers.options['abstol'] = float(1e-7)
    >>> cvxopt.solvers.options['reltol'] = float(1e-7)
    >>> vals = get_many_values(S)
    >>> ', '.join([ '%s: %.5f' % (k, vals[k]) for k in sorted(vals.keys()) ])
    'lovasz: 3.80697, schrijver(hermit): 3.31461, schrijver(ppt): 2.42820, schrijver(psd&ppt): 2.42820, schrijver(psd): 3.31461, szegedy(hermit): 4.55314, szegedy(ppt): inf, szegedy(psd&ppt): inf, szegedy(psd): 4.55314'
    """

    if which_ones is None:
        which_ones = [
            'lovasz',
            'schrijver(hermit)',
            'schrijver(ppt)',
            'schrijver(psd&ppt)',
            'schrijver(psd)',
            'szegedy(hermit)',
            'szegedy(ppt)',
            'szegedy(psd&ppt)',
            'szegedy(psd)',
        ]

    ret = {}
    if 'lovasz' in which_ones:
        ret['lovasz'] = lovasz_theta(S)
    for cone in ('hermit', 'ppt', 'psd', 'psd&ppt'):
        if 'schrijver('+cone+')' in which_ones:
            ret['schrijver('+cone+')'] = schrijver(S, cone)
        if 'szegedy('+cone+')' in which_ones:
            ret['szegedy('+cone+')'] = szegedy(S, cone)
    return ret

### Validation code ####################

def test_lovasz(S):
    """
    >>> ha = qudit('a', 3)
    >>> np.random.seed(1)
    >>> S = TensorSubspace.create_random_hermitian(ha, 5, tracefree=True).perp()
    >>> test_lovasz(S)
    t: 3.8069736
    total err: 0.0000000
    total err: 0.0000000
    duality gap: 0.0000000
    """

    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = float(1e-8)
    cvxopt.solvers.options['reltol'] = float(1e-8)

    info = lovasz_theta(S, True)
    assert info['sdp_stats']['status'] == 'optimal'
    print('t: %.7f' % info['t'])
    (tp, errp) = check_lovasz_primal(S, *[ info[x] for x in 'rho,T'.split(',') ])
    (td, errd) = check_lovasz_dual(S, *[ info[x] for x in 'Y'.split(',') ])
    print('duality gap: %.7f' % (td-tp))

def test_schrijver(S, cones=('hermit', 'psd', 'ppt', 'psd&ppt')):
    """
    >>> ha = qudit('a', 3)
    >>> np.random.seed(1)
    >>> S = TensorSubspace.create_random_hermitian(ha, 5, tracefree=True).perp()
    >>> test_schrijver(S)
    --- Schrijver with hermit
    t: 3.3146051
    total err: 0.0000000
    total err: 0.0000000
    duality gap: 0.0000000
    --- Schrijver with psd
    t: 3.3146051
    total err: 0.0000000
    total err: 0.0000000
    duality gap: 0.0000000
    --- Schrijver with ppt
    t: 2.4281982
    total err: 0.0000000
    total err: 0.0000000
    duality gap: -0.0000000
    --- Schrijver with psd&ppt
    t: 2.4281982
    total err: 0.0000000
    total err: 0.0000000
    duality gap: -0.0000000
    """

    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = float(1e-8)
    cvxopt.solvers.options['reltol'] = float(1e-8)

    # FIXME - are 'ppt' and 'psd&ppt' always the same value?
    for cone in cones:
        print('--- Schrijver with', cone)
        info = schrijver(S, cone, True)
        assert info['sdp_stats']['status'] == 'optimal'
        print('t: %.7f' % info['t'])
        (tp, errp) = check_schrijver_primal(S, cone, *[ info[x] for x in 'rho,T'.split(',') ])
        (td, errd) = check_schrijver_dual(S, cone, *[ info[x] for x in 'Y,L_map'.split(',') ])
        print('duality gap: %.7f' % (td-tp))

def test_szegedy(S):
    """
    >>> ha = qudit('a', 3)
    >>> np.random.seed(1)
    >>> S = TensorSubspace.create_random_hermitian(ha, 5, tracefree=True).perp()
    >>> test_szegedy(S)
    --- Szegedy with hermit
    t: 4.5531383
    total err: 0.0000000
    total err: 0.0000000
    duality gap: 0.0000000
    --- Szegedy with psd
    t: 4.5531383
    total err: 0.0000000
    total err: 0.0000000
    duality gap: -0.0000000
    --- Szegedy with ppt
    t: inf
    total err: 0.0000000
    --- Szegedy with psd&ppt
    t: inf
    total err: 0.0000000
    """

    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = float(1e-8)
    cvxopt.solvers.options['reltol'] = float(1e-8)

    # FIXME - are 'ppt' and 'psd&ppt' always the same value?
    for cone in ('hermit', 'psd', 'ppt', 'psd&ppt'):
        print('--- Szegedy with', cone)
        info = szegedy(S, cone, True)
        assert info['sdp_stats']['status'] in ['optimal', 'primal infeasible']
        print('t: %.7f' % info['t'])
        is_opt = (info['sdp_stats']['status'] == 'optimal')
        info['is_opt'] = is_opt
        (tp, errp) = check_szegedy_primal(S, cone, *[ info[x] for x in 'rho,T,L_map,is_opt'.split(',') ])
        if is_opt:
            (td, errd) = check_szegedy_dual(S, cone, *[ info[x] for x in 'Y'.split(',') ])
            print('duality gap: %.7f' % (td-tp))

def check_lovasz_primal(S, rho, T, report=True):
    r"""
    Verify Lovasz primal solution.
    Returns ``(t, err)`` where ``t`` is the value and ``err`` is the amount by which
    feasibility constrains are violated.
    """

    return checking_routine(S, None, { 'lovasz_primal': (rho, T) }, report)

def check_lovasz_dual(S, Y, report=True):
    r"""
    Verify Lovasz dual solution.
    Returns ``(t, err)`` where ``t`` is the value and ``err`` is the amount by which
    feasibility constrains are violated.
    """

    return checking_routine(S, None, { 'lovasz_dual': (Y,) }, report)

def check_schrijver_primal(S, cones, rho, T, report=True):
    r"""
    Verify Schrijver primal solution.
    Returns ``(t, err)`` where ``t`` is the value and ``err`` is the amount by which
    feasibility constrains are violated.
    """

    return checking_routine(S, cones, { 'schrijver_primal': (rho, T) }, report)

def check_schrijver_dual(S, cones, Y, L_map, report=True):
    r"""
    Verify Schrijver dual solution.
    Returns ``(t, err)`` where ``t`` is the value and ``err`` is the amount by which
    feasibility constrains are violated.
    """

    return checking_routine(S, cones, { 'schrijver_dual': (Y, L_map) }, report)

def check_szegedy_primal(S, cones, rho, T, L_map, is_opt, report=True):
    r"""
    Verify Schrijver primal solution.
    Returns ``(t, err)`` where ``t`` is the value and ``err`` is the amount by which
    feasibility constrains are violated.
    """

    return checking_routine(S, cones, { 'szegedy_primal': (rho, T, L_map, is_opt) }, report)

def check_szegedy_dual(S, cones, Y, report=True):
    r"""
    Verify Szegedy dual solution.
    Returns ``(t, err)`` where ``t`` is the value and ``err`` is the amount by which
    feasibility constrains are violated.
    """

    return checking_routine(S, cones, { 'szegedy_dual': Y }, report)

def checking_routine(S, cones, task, report):
    ncg = GraphProperties(S)

    if cones is not None:
        cones = ncg.get_cone_set(cones)
        cone_names = frozenset(C['name'] for C in cones)

    ha = ncg.top_space
    hb = ncg.bottom_space
    R = ncg.R
    ddag = ncg.ddag

    Sb = S.map(lambda x: x.relabel({ ha: hb, ha.H: hb.H }).conj())

    Phi = ha.eye().relabel({ ha.H: hb })
    J = Phi.O

    def proj_Sp_ot_all(x):
        ret = (ha*hb).O.array()
        for pa in S.perp():
            ret += pa * (x * pa.H).trace(ha)
        return ret

    def proj_Sp_ot_Sp(x):
        ret = (ha*hb).O.array()
        for pa in S.perp():
            foo = (x * pa.H).trace(ha)
            ret += pa * Sb.perp().project(foo)
        return ret

    def proj_S_ot_S(x):
        ret = (ha*hb).O.array()
        for pa in S:
            foo = (x * pa.H).trace(ha)
            ret += pa * Sb.project(foo)
        return ret

    err = {}

    if 'lovasz_primal' in task:
        (rho, T) = task['lovasz_primal']

        assert rho.space == hb.O
        assert T.space == (ha*hb).O
        val = 1 + (Phi.H * T * Phi).real
        err[r'trace(rho)'] = abs(1 - rho.trace())
        err[r'rho PSD'] = check_psd(rho)
        err[r'T + I \ot rho PSD'] = check_psd(T + ha.eye()*rho)

        err[r"T \in S^\perp \ot \linop{A'}"] = (T - proj_Sp_ot_all(T)).norm()

    if 'lovasz_dual' in task:
        (Y,) = task['lovasz_dual']

        val = Y.trace(ha).eigvalsh()[-1]

        err[r'Y \succeq J'] = check_psd(Y - J)

        err[r"Y \perp S^\perp \ot \linop{A'}"] = proj_Sp_ot_all(Y).norm()

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

        err[r'T \in S^\perp \ot \bar{S}^\perp'] = (T - proj_Sp_ot_Sp(T)).norm()

    if 'schrijver_dual' in task:
        (Y, L_map) = task['schrijver_dual']

        val = Y.trace(ha).eigvalsh()[-1]

        err[r'Y \succeq J'] = check_psd(Y - J)

        L = np.sum(list(L_map.values()))

        err[r'Y+L+L^\dag \perp S^\perp \ot \bar{S}^\perp'] = proj_Sp_ot_Sp(Y+L+L.H).norm()

        for C in cone_names:
            L = L_map[C]
            M = R(L) + R(L).H
            if C == 'psd':
                err[r'L_PSD'] = check_psd(M)
            elif C == 'ppt':
                err[r'L_PPT'] = check_psd(M.transpose(ha))
            else:
                assert 0

    if 'szegedy_primal' in task:
        (rho, T, L_map, is_opt) = task['szegedy_primal']

        assert rho.space == hb.O
        assert T.space == (ha*hb).O

        if is_opt:
            val = 1 + (Phi.H * T * Phi).real
            err[r'trace(rho)'] = abs(1 - rho.trace())
        else:
            # Certificate of primal infeasibility
            val = np.inf
            err[r'trace(rho)'] = abs(rho.trace())
            err[r'<Phi|T|Phi>=1'] = abs(1-(Phi.H * T * Phi).real)

        err[r'rho PSD'] = check_psd(rho)
        err[r'T + I \ot rho PSD'] = check_psd(T + ha.eye()*rho)

        L = np.sum(list(L_map.values()))

        err[r'T+L+L^\dag \perp S \ot \bar{S}'] = proj_S_ot_S(T+L+L.H).norm()

        for C in cone_names:
            L = L_map[C]
            M = R(L) + R(L).H
            if C == 'psd':
                err[r'L_PSD'] = check_psd(M)
            elif C == 'ppt':
                err[r'L_PPT'] = check_psd(M.transpose(ha))
            else:
                assert 0

    if 'szegedy_dual' in task:
        Y = task['szegedy_dual']

        val = Y.trace(ha).eigvalsh()[-1]
        err[r'Y \succeq J'] = check_psd(Y - J)
        err[r'Y^\ddag - Y'] = (Y - ddag(Y)).norm()

        for C in cone_names:
            if C == 'psd':
                err[r'T_PSD'] = check_psd(R(Y))
            elif C == 'ppt':
                err[r'T_PPT'] = check_psd(R(Y).transpose(ha))
            else:
                assert 0

        err[r'Y \in S \ot \bar{S}'] = (Y - proj_S_ot_S(Y)).norm()

    ### Tally and report

    assert min(err.values()) >= 0

    if report:
        for (k, v) in err.items():
            if v > 1e-7:
                print('err[%s] = %g' % (k, v))

        print('total err: %.7f' % sum(err.values()))

    return (val, err)

# If this module is run from the command line, run the doctests.
if __name__ == "__main__":
    # Doctests require not getting progress messages from SDP solver.
    cvxopt.solvers.options['show_progress'] = False

    print("Running doctests.")
    import doctest
    doctest.testmod()
