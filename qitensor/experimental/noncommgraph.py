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

def make_F_real(Fx_list, F0_list):
    '''
    Convert F0, Fx arrays to real if needed, by considering C as a vector space
    over R.  This is needed because cvxopt cannot handle complex inputs.
    '''

    def mat_cplx_to_real(cmat):
        rmat = np.zeros((cmat.shape[0], 2, cmat.shape[1], 2))
        rmat[:, 0, :, 0] = cmat.real
        rmat[:, 1, :, 1] = cmat.real
        rmat[:, 0, :, 1] = -cmat.imag
        rmat[:, 1, :, 0] = cmat.imag
        return rmat.reshape(cmat.shape[0]*2, cmat.shape[1]*2)

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

    # Note: Fx and F0 must be negated when passed to cvxopt.sdp.
    (Fx_list, F0_list) = make_F_real(Fx_list, F0_list)
    Gs = [cvxopt.base.matrix(-Fx.reshape(Fx.shape[0]**2, Fx.shape[2])) for Fx in Fx_list]
    hs = [cvxopt.base.matrix(-F0) for F0 in F0_list]

    sol = cvxopt.solvers.sdp(cvxopt.base.matrix(c), Gs=Gs, hs=hs)
    xvec = np.array(sol['x']).flatten()
    return (xvec, sol)

### The main code ######################

class NoncommutativeGraph(object):
    """Non-commutative graphs as described in arXiv:1002.2514."""

    def __init__(self, S):
        """
        Create a non-commutative graph from provided TensorSubspace.
        """

        self.S_basis = np.array(S.basis())
        self.Sp_basis = np.array(S.perp().basis())

        self._validate_S_Sp(self.S_basis, self.Sp_basis)

        self.S = S

    @classmethod
    def _validate_S_Sp(cls, S, Sp):
        '''
        Verify that (S, Sp) actually correspond to a non-commutative graph.

        S must be a Hermitian orthonormal basis, and Sp must be an orthonormal basis
        for the orthogonal complement of S.
        '''

        tol = 1e-12

        assert len(S.shape) == 3
        assert len(Sp.shape) == 3
        n = S.shape[1]
        assert S.shape[2] == n
        assert Sp.shape[1] == n
        assert Sp.shape[2] == n
        assert S.shape[0] + Sp.shape[0] == n*n

        foo = np.tensordot(S, Sp.conjugate(), axes=[[1,2],[1,2]])
        if linalg.norm(foo) > tol:
            raise ArithmeticError("S and Sp are not orthogonal")

        foo = np.tensordot(S, S.conjugate(), axes=[[1,2],[1,2]])
        if linalg.norm(foo - np.eye(S.shape[0])) > tol:
            raise ArithmeticError("S is not orthonormal")

        foo = np.tensordot(Sp, Sp.conjugate(), axes=[[1,2],[1,2]])
        if linalg.norm(foo - np.eye(Sp.shape[0])) > tol:
            raise ArithmeticError("S is not orthonormal")

        if S.shape[0] + Sp.shape[0] != n*n:
            raise ArithmeticError("S+Sp is not a complete basis")

        foo = np.trace(Sp, axis1=1, axis2=2)
        if linalg.norm(foo) > tol:
            raise ArithmeticError("S does not contain I (Sp is not trace free)")

        foo = np.tensordot(S, Sp, axes=[[2,1],[1,2]])
        if linalg.norm(foo) > tol:
            raise ArithmeticError("S is not Hermitian")

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
        >>> theta = G.lovasz()
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

    def _get_Y_basis(self):
        """
        Compute a basis for the allowed Y operators for Theorem 9 of
        arXiv:1002.2514.  These are the operators which are Hermitian and are
        in S*L(A').  Note that this basis is intended to map real vectors to
        complex Y operators.
        """

        (nS, n, _n) = self.S_basis.shape
        assert n == _n

        # FIXME - not the fastest way
        foo = self.S.tensor_prod(TensorSubspace.full((n, n)))
        bar = TensorSubspace(
            np.array(foo.       basis()).transpose((0, 1, 3, 2, 4)) \
                    .reshape(foo.dim(), n*n, n*n),
            np.array(foo.perp().basis()).transpose((0, 1, 3, 2, 4)) \
                    .reshape(foo.perp().dim(), n*n, n*n),
            1e-10, None
        )
        baz = bar.hermitian_basis()
        ret = baz.reshape(baz.shape[0], n, n, n, n).transpose((1, 2, 3, 4, 0))
        # [ |a>, |a'>, <a|, <a'| ; idx ]
        return ret

    # FIXME - is this one faster?
    def _get_Y_basis_old(self):
        """
        Compute a basis for the allowed Y operators for Theorem 9 of
        arXiv:1002.2514.  These are the operators which are Hermitian and are
        in S*L(A').  Note that this basis is intended to map real vectors to
        complex Y operators.
        """

        (nS, n, _n) = self.S_basis.shape
        assert n == _n

        # x_to_Y = [ |a>, |a'>, <a|, <a'| ; Si, |a'>, <a'| ]
        x_to_Y = np.zeros((n,n, n,n, nS,n,n), dtype=complex)
        for (S_i, Sv) in enumerate(self.S_basis):
            for (Ap_i, Ap_j) in itertools.product(range(n), repeat=2):
                x_to_Y[:, Ap_i, :, Ap_j, S_i, Ap_i, Ap_j] = Sv
        x_to_Y = x_to_Y.reshape(n,n, n,n, nS,n**2)
        # project onto Hermitian space while simulating complex values with
        # reals on the x side
        x_to_Y_1  = x_to_Y    + np.transpose(x_to_Y,    [2, 3, 0, 1, 4, 5]).conjugate()
        x_to_Y_1j = x_to_Y*1j + np.transpose(x_to_Y*1j, [2, 3, 0, 1, 4, 5]).conjugate()
        x_to_Y = np.concatenate((
                x_to_Y_1 .reshape(n,n,n,n, nS*(n**2)),
                x_to_Y_1j.reshape(n,n,n,n, nS*(n**2))
            ), axis=4)

        # decrease parameters by only taking linearly independent subspace
        sqrmat = np.array([x_to_Y.real, x_to_Y.imag]).reshape(2*(n**4), 2*nS*(n**2))
        (U, s, V) = linalg.svd(sqrmat)
        n_indep = np.sum(s > 1e-10)
        Y_basis = U[:, :n_indep]
        x_to_Y_reduced_real = Y_basis.reshape(2,n,n,n,n, n_indep)
        x_to_Y_reduced = x_to_Y_reduced_real[0] + 1j*x_to_Y_reduced_real[1]

        for i in xrange(n_indep):
            s = x_to_Y_reduced[:, :, :, :, i]
            sH = np.transpose(s, [2, 3, 0, 1]).conjugate()
            assert linalg.norm(s - sH) < 1e-13
            # correct for numerical error and make it exactly Hermitian
            x_to_Y_reduced[:, :, :, :, i] = (s + sH)/2

        return x_to_Y_reduced

    def lovasz(self, long_return=False):
        """
        Compute the non-commutative generalization of the Lovasz function,
        using Theorem 9 of arXiv:1002.2514.

        If the long_return option is True, then some extra status and internal
        quantities are returned (such as the optimal Y operator).
        """

        (nS, n, _n) = self.S_basis.shape
        assert n == _n

        Y_basis = self._get_Y_basis()
        # x = [t, Y.A:Si * Y.A':i * Y.A':j]
        xvec_len = 1 + Y_basis.shape[4]
        x_to_Y = np.concatenate((
                np.zeros((n,n,n,n, 1)),
                Y_basis
            ), axis=4)
        assert x_to_Y.shape[4] == xvec_len

        phi_phi = np.zeros((n,n, n,n), dtype=complex)
        for (i, j) in itertools.product(range(n), repeat=2):
            phi_phi[i, i, j, j] = 1
        phi_phi = phi_phi.reshape(n**2, n**2)

        # Cost vector.
        # x = [t, Y.A:Si * Y.A':i * Y.A':j]
        c = np.zeros(xvec_len)
        c[0] = 1

        # tI - tr_A{Y} >= 0
        Fx_1 = -np.trace(x_to_Y, axis1=0, axis2=2)
        for i in xrange(n):
            Fx_1[i, i, 0] = 1

        F0_1 = np.zeros((n, n))

        # Y - |phi><phi| >= 0
        Fx_2 = x_to_Y.reshape(n**2, n**2, xvec_len)
        F0_2 = phi_phi

        (xvec, sdp_stats) = call_sdp(c, (Fx_1, Fx_2), (F0_1, F0_2))
        if sdp_stats['status'] != 'optimal':
            raise ArithmeticError(sdp_stats['status'])

        t = xvec[0]
        Y = np.dot(x_to_Y, xvec)

        # some sanity checks to make sure the output makes sense
        verify_tol=1e-7
        if verify_tol:
            err = linalg.eigvalsh(np.dot(Fx_1, xvec).reshape(n, n)   - F0_1)[0] > -verify_tol
            if err < -verify_tol: print "WARNING: F1 err =", err
            err = linalg.eigvalsh(np.dot(Fx_2, xvec).reshape(n**2, n**2) - F0_2)[0]
            if err < -verify_tol: print "WARNING: F2 err =", err
            err = linalg.eigvalsh(Y.reshape(n**2, n**2) - phi_phi)[0]
            if err < -verify_tol: print "WARNING: phi_phi err =", err
            maxeig = linalg.eigvalsh(np.trace(Y, axis1=0, axis2=2))[-1].real
            err = abs(np.dot(c, xvec) - maxeig)
            if err > verify_tol: print "WARNING: t err =", err

            # make sure it is in S*L(A')
            for mat in self.Sp_basis:
                dp = np.tensordot(Y, mat.conjugate(), axes=[[0, 2], [0, 1]])
                err = linalg.norm(dp)
                if err > 1e-13: print "err:", err
                assert err < 1e-13

        if long_return:
            ret = {}
            for key in ['n', 'x_to_Y', 'Fx_1', 'Fx_2', 'F0_1', 'F0_2', 'phi_phi', 'c', 't', 'Y', 'xvec', 'sdp_stats']:
                ret[key] = locals()[key]
            return ret
        else:
            return t

# Maybe this cannot be computed using a semidefinite program.
#
#    def small_lovasz(self, long_return=False):
#        """
#        Compute the non-commutative generalization of the Lovasz function
#        (non-multiplicative version), using Eq. 5 of arXiv:1002.2514.
#
#        If the long_return option is True, then some extra status and internal
#        quantities are returned.
#        """
#
#        (nSp, n, _n) = self.Sp_basis.shape
#        assert n == _n
#
#        Sp_basis = self.S.perp().hermitian_basis()
#        # x = [t, T:Si]
#        xvec_len = 1 + Sp_basis.shape[2]
#        x_to_Sp = np.concatenate((
#                np.zeros((n,n, 1)),
#                Sp_basis
#            ), axis=2)
#        assert x_to_Sp.shape[2] == xvec_len
#
#        # Cost vector.
#        c = np.zeros(xvec_len)
#        c[0] = -1
#
#        # FIXME - any way to maximize the max eigenvalue?
#        # tI - T >= 0
#        Fx_1 = -x_to_Sp
#        for i in xrange(n):
#            Fx_1[i, i, 0] = 1
#
#        F0_1 = np.zeros((n, n))
#
#        # T + I >= 0
#        Fx_2 = x_to_Sp
#        F0_2 = -np.eye(n)
#
#        (xvec, sdp_stats) = call_sdp(c, (Fx_1, Fx_2), (F0_1, F0_2))
#        if sdp_stats['status'] != 'optimal':
#            raise ArithmeticError(sdp_stats['status'])
#
#        t = xvec[0]
#        T = np.dot(x_to_Sp, xvec)
#
#        theta = t+1
#
#        if long_return:
#            ret = {}
#            for key in ['n', 'x_to_Sp', 'Fx_1', 'Fx_2', 'F0_1', 'F0_2', 'c', 't', 'theta', 'T', 'xvec', 'sdp_stats']:
#                ret[key] = locals()[key]
#            return ret
#        else:
#            return theta

# If this module is run from the command line, run the doctests.
#if __name__ == "__main__":
#    # Doctests require not getting progress messages from SDP solver.
#    cvxopt.solvers.options['show_progress'] = False
#
#    print "Running doctests."
#
#    import doctest
#    doctest.testmod()
