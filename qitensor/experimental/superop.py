import numpy as np
import itertools

from qitensor import qudit, NotKetSpaceError, HilbertArray

# FIXME - use exceptions rather than assert

class Superoperator(object):
    def __init__(self, ha, hb, hc, J, check_tp=True):
        self.ha = self._to_ket_space(ha)
        self.hb = self._to_ket_space(hb)
        self.hc = self._to_ket_space(hc)

        assert J.space == self.hb * self.hc * self.ha.H

        if check_tp and (J.H*J - ha.eye()).norm() > 1e-14:
            raise ValueError('channel is not trace preserving')

        self.J = J

    @classmethod
    def _make_environ_spc(cls, espc_def, field, dim):
        if espc_def is None:
            espc_def = 'environ_'+str(np.random.randint(1e12))
        return qudit(espc_def, dim, dtype=field)

    @classmethod
    def _to_ket_space(cls, spc):
        if not spc.bra_set:
            return spc
        if not spc.ket_set:
            return spc.H
        if spc == spc.O:
            return spc.ket_space
        raise NotKetSpaceError('need a bra, ket, or self-adjoint space, not '+str(spc))

    def __str__(self):
        return '<Superoperator( '+str(self.ha.O)+' to '+str(self.hb.O)+' )>'

    def __repr__(self):
        return str(self)

    def __call__(self, rho):
        assert rho.space.bra_ket_set >= self.ha.O.bra_ket_set
        return (self.J * rho * self.J.H).trace(self.hc)

    def __mul__(self, other):
        if isinstance(other, Superoperator):
            raise NotImplementedError() # FIXME

        if isinstance(other, HilbertArray):
            raise ValueError()

        # hopefully `other` is a scalar
        s = self.ha.base_field.sqrt(other)
        return Superoperator(self.ha, self.hb, self.hc, self.J*s, check_tp=False)

    def __add__(self, other):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qudit('a', 4)
        >>> hb = qudit('b', 3)
        >>> E1 = Superoperator.random(ha, hb)
        >>> E2 = Superoperator.random(ha, hb)
        >>> rho = ha.random_density()
        >>> chi = (E1*0.2 + E2*0.8)(rho)
        >>> xi  = E1(rho)*0.2 + E2(rho)*0.8
        >>> (chi - xi).norm() < 1e-14
        True
        """

        if not isinstance(other, Superoperator):
            raise ValueError()

        assert self.ha == other.ha
        assert self.hb == other.hb

        m = self.as_matrix() + other.as_matrix()
        return Superoperator.from_matrix(m, self.ha, self.hb, check_tp=False)

    def compl(self):
        # FIXME - cache this (circularly)
        return Superoperator(self.ha, self.hc, self.hb, self.J, check_tp=False)

    def as_four_tensor(self):
        # FIXME - untested
        da = self.ha.dim()
        db = self.hb.dim()
        ret = np.zeros((db, da, db, da), dtype=self.ha.base_field.dtype)
        for j in self.hc.index_iter():
            op = self.J[{ self.hc: j }].as_np_matrix(row_space=self.ha.H)
            ret += np.tensordot(op, op.conj(), axes=([],[]))
        return ret

    def as_matrix(self):
        da = self.ha.dim()
        db = self.hb.dim()
        t = self.as_four_tensor()
        t = t.transpose([0,2,1,3])
        t = t.reshape(db**2, da**2)
        return np.matrix(t)

    @classmethod
    def from_matrix(cls, m, spc_in, spc_out, espc_def=None):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qudit('a', 2)
        >>> hb = qudit('b', 3)
        >>> hx = qudit('x', 5)
        >>> E1 = Superoperator.random(ha*hb, hx)
        >>> E2 = Superoperator.random(hx, ha*hb)
        >>> m = E2.as_matrix() * E1.as_matrix()
        >>> E3 = Superoperator.from_matrix(m, ha*hb, ha*hb)
        >>> rho = (ha*hb).random_density()
        >>> (E2(E1(rho)) - E3(rho)).norm() < 1e-14
        True
        """

        ha = cls._to_ket_space(spc_in)
        hb = cls._to_ket_space(spc_out)
        da = ha.dim()
        db = hb.dim()
        t = np.array(m)
        assert t.shape == (db*db, da*da)
        t = t.reshape(db, db, da, da)
        t = t.transpose([0, 2, 1, 3])
        t = t.reshape(db*da, db*da)

        field = ha.base_field

        if field.mat_norm(np.transpose(np.conj(t)) - t) > 1e-14:
            raise ValueError("matrix didn't correspond to a totally positive superoperator")

        (ew, ev) = field.mat_eig(t, True)

        if np.min(ew) < 0:
            raise ValueError("matrix didn't correspond to a totally positive superoperator")

        hc = cls._make_environ_spc(espc_def, ha.base_field, da*db)

        J = (hb * hc * ha.H).array()

        for i in range(da*db):
            J[{ hc: i }] = (hb * ha.H).array(ev[:,i] * np.sqrt(ew[i]), reshape=True)

        return Superoperator(ha, hb, hc, J)

    @classmethod
    def convex_combination(cls, Elist, Plist, espc_def=None):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qudit('a', 4)
        >>> hb = qudit('b', 3)
        >>> E1 = Superoperator.random(ha, hb)
        >>> E2 = Superoperator.random(ha, hb)
        >>> E = Superoperator.convex_combination((E1, E2), (0.2, 0.8))
        >>> rho = ha.random_density()
        >>> (E(rho) - (E1(rho)*0.2 + E2(rho)*0.8)).norm() < 1e-14
        True
        """

        assert len(Elist) == len(Plist)
        assert np.abs(np.sum(Plist) - 1) < 1e-14

        E0 = Elist[0]

        for E in Elist:
            assert E.ha == E0.ha
            assert E.hb == E0.hb

        m = np.sum([ E.as_matrix()*p for (E, p) in zip(Elist, Plist) ], axis=0)

        return cls.from_matrix(m, E0.ha, E0.hb, espc_def)

    @classmethod
    def random(cls, spc_in, spc_out, espc_def=None):
        ha = cls._to_ket_space(spc_in)
        hb = cls._to_ket_space(spc_out)
        dc = ha.dim() * hb.dim()
        hc = cls._make_environ_spc(espc_def, ha.base_field, dc)
        J = (hb*hc*ha.H).random_isometry()
        return cls(ha, hb, hc, J)

    @classmethod
    def unitary(cls, U, espc_def=None):
        """
        >>> from qitensor import qubit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> U = ha.random_unitary()
        >>> rho = (ha*hb).random_density()
        >>> E = Superoperator.unitary(U)
        >>> (E(rho) - U*rho*U.H).norm() < 1e-14
        True
        """

        ha = U.space.bra_space().H
        hb = U.space.ket_space()
        hc = cls._make_environ_spc(espc_def, ha.base_field, 1)
        J = U * hc.ket(0)
        return cls(ha, hb, hc, J)

    @classmethod
    def identity(cls, spc, espc_def=None):
        """
        >>> from qitensor import qubit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> rho = (ha*hb).random_density()
        >>> E = Superoperator.identity(ha)
        >>> (E(rho) - rho).norm() < 1e-14
        True
        """

        return cls.unitary(spc.eye(), espc_def)

    @classmethod
    def totally_noisy(cls, spc, espc_def=None):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qudit('a', 5)
        >>> rho = ha.random_density()
        >>> E = Superoperator.totally_noisy(ha)
        >>> (E(rho) - ha.eye()/ha.dim()).norm() < 1e-14
        True
        """

        # FIXME - check definition and name
        ha = cls._to_ket_space(spc)
        d = ha.dim()
        d2 = d*d
        hc = cls._make_environ_spc(espc_def, ha.base_field, d2)
        J = (ha.O*hc).array()
        for (i, (j, k)) in enumerate(itertools.product(ha.index_iter(), repeat=2)):
            J[{ ha.H: j, ha: k, hc: i }] = 1
        J /= ha.base_field.sqrt(d)
        return cls(ha, ha, hc, J)

    @classmethod
    def noisy(cls, spc, p, espc_def=None):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qudit('a', 5)
        >>> rho = ha.random_density()
        >>> E = Superoperator.noisy(ha, 0.2)
        >>> (E(rho) - 0.8*rho - 0.2*ha.eye()/ha.dim()).norm() < 1e-14
        True
        """

        # FIXME - check definition and name
        assert 0 <= p <= 1
        E0 = cls.totally_noisy(spc)
        E1 = cls.identity(spc)
        return cls.convex_combination((E0, E1), (p, 1-p))

ha = qudit('a', 2)
rho = ha.random_density()
E = Superoperator.noisy(ha, 0.2)
print rho
print E(rho)
