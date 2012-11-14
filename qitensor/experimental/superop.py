import numpy as np
import itertools

from qitensor import qudit, NotKetSpaceError, \
    HilbertSpace, HilbertArray, HilbertError

toler = 1e-12

# FIXME - use exceptions rather than assert
# FIXME - pickling

__all__ = ['Superoperator', 'CP_Map']

class Superoperator(object):
    def __init__(self, ha, hb, m):
        self.ha = self._to_ket_space(ha)
        self.hb = self._to_ket_space(hb)
        self._m = np.matrix(m)

        assert m.shape == (self.hb.O.dim(), self.ha.O.dim())

    @classmethod
    def _make_environ_spc(cls, espc_def, field, dim):
        if espc_def is None:
            espc_def = 'environ_'+str(np.random.randint(1e12))

        if isinstance(espc_def, HilbertSpace):
            if espc_def.dim() < dim:
                raise HilbertError('environment space not big enough: %d vs %d'
                    % (espc_def.dim(), dim))
            return espc_def

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
        return 'Superoperator( '+str(self.ha.O)+' to '+str(self.hb.O)+' )'

    def __repr__(self):
        return str(self)

    def as_matrix(self):
        return self._m

    def __call__(self, rho):
        assert rho.space.bra_ket_set >= self.ha.O.bra_ket_set
        (row_space, col_space) = rho._get_row_col_spaces(col_space=self.ha.O)
        ret_vec = self._m * rho.as_np_matrix(col_space=self.ha.O)
        if len(row_space):
            out_space = self.hb.O * np.prod(row_space)
        else:
            out_space = self.hb.O
        return out_space.array(ret_vec, reshape=True, input_axes=self.hb.O.axes+row_space)

    def __mul__(self, other):
        if isinstance(other, Superoperator):
            raise NotImplementedError() # FIXME

        if isinstance(other, HilbertArray):
            raise ValueError()

        # hopefully `other` is a scalar
        return Superoperator(self.ha, self.hb, self._m*other)

    def __rmul__(self, other):
        # hopefully `other` is a scalar
        return self * other

    def __add__(self, other):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qudit('a', 4)
        >>> hb = qudit('b', 3)
        >>> E1 = CP_Map.random(ha, hb)
        >>> E2 = CP_Map.random(ha, hb)
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

        return Superoperator(self.ha, self.hb, self._m + other._m)

    @classmethod
    def from_function(cls, ha, f):
        """
        >>> from qitensor import qubit, qudit
        >>> from qitensor.experimental.superop import Superoperator, CP_Map
        >>> ha = qudit('a', 3)
        >>> hb = qubit('b')
        >>> E = Superoperator.from_function(ha, lambda rho: rho.T)
        >>> rho = (ha*hb).random_density()
        >>> (E(rho) - rho.transpose(ha)).norm() < 1e-14
        True

        >>> Superoperator.from_function(ha, lambda rho: rho.H)
        Traceback (most recent call last):
            ...
        ValueError: function was not linear

        >>> CP_Map.from_function(ha, lambda rho: rho.T)
        Traceback (most recent call last):
            ...
        ValueError: matrix didn't correspond to a totally positive superoperator (min eig=-1.0)
        """

        ha = cls._to_ket_space(ha)
        hb = f(ha.eye()).space
        assert hb == hb.H
        hb = hb.ket_space()

        m = np.zeros((hb.dim()**2, ha.dim()**2), ha.base_field.dtype)
        for (i, x) in enumerate(ha.O.index_iter()):
            m[:, i] = f(ha.O.basis_vec(x)).nparray.flatten()

        E = Superoperator(ha, hb, m)

        rho = ha.random_density()
        if (E(rho) - f(rho)).norm() > toler:
            raise ValueError('function was not linear')

        if cls == CP_Map:
            E = E.upgrade_to_cp_map()

        return E

    def upgrade_to_cp_map(self, espc_def=None, check_tp=True):
        return CP_Map.from_matrix(self._m, self.ha, self.hb, espc_def=espc_def, check_tp=check_tp)

#    @classmethod
#    def convex_combination(cls, Elist, Plist, espc_def=None):
#        """
#        >>> from qitensor import qudit
#        >>> from qitensor.experimental.superop import Superoperator
#        >>> ha = qudit('a', 4)
#        >>> hb = qudit('b', 3)
#        >>> E1 = CP_Map.random(ha, hb)
#        >>> E2 = CP_Map.random(ha, hb)
#        >>> E = Superoperator.convex_combination((E1, E2), (0.2, 0.8))
#        >>> rho = ha.random_density()
#        >>> (E(rho) - (E1(rho)*0.2 + E2(rho)*0.8)).norm() < 1e-14
#        True
#        """
#
#        assert len(Elist) == len(Plist)
#        assert np.abs(np.sum(Plist) - 1) < toler
#
#        E0 = Elist[0]
#
#        for E in Elist:
#            assert E.ha == E0.ha
#            assert E.hb == E0.hb
#
#        m = np.sum([ E._m*p for (E, p) in zip(Elist, Plist) ], axis=0)
#
#        return Superoperator(E0.ha, E0.hb, m)

class CP_Map(Superoperator):
    def __init__(self, ha, hb, hc, J, check_tp=True, _complimentary_channel=None):
        """
        >>> ha = qudit('a', 2)
        >>> hb = qudit('b', 2)
        >>> hd = qudit('d', 3)
        >>> rho = (ha*hd).random_density()
        >>> E = CP_Map.random(ha, hb)
        >>> ((E.J * rho * E.J.H).trace(E.hc) - E(rho)).norm() < 1e-14
        True
        """

        ha = self._to_ket_space(ha)
        hb = self._to_ket_space(hb)
        hc = self._to_ket_space(hc)

        assert J.space == hb * hc * ha.H

        if check_tp and (J.H*J - ha.eye()).norm() > toler:
            raise ValueError('channel is not trace preserving')

        da = ha.dim()
        db = hb.dim()
        t = np.zeros((db, da, db, da), dtype=ha.base_field.dtype)
        for j in hc.index_iter():
            op = J[{ hc: j }].as_np_matrix(row_space=ha.H)
            t += np.tensordot(op, op.conj(), axes=([],[]))
        t = t.transpose([0,2,1,3])
        t = t.reshape(db**2, da**2)
        t = np.matrix(t)

        super(CP_Map, self).__init__(ha, hb, t)

        self.J = J
        self.hc = hc

        if _complimentary_channel is None:
            self.C = CP_Map(self.ha, self.hc, self.hb, self.J, check_tp=False, \
                _complimentary_channel=self)
        else:
            self.C = _complimentary_channel

    def __str__(self):
        return 'CP_Map( '+str(self.ha.O)+' to '+str(self.hb.O)+' )'

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        if isinstance(other, Superoperator):
            raise NotImplementedError() # FIXME

        if isinstance(other, HilbertArray):
            raise ValueError()

        # hopefully `other` is a scalar
        s = self.ha.base_field.sqrt(other)
        return CP_Map(self.ha, self.hb, self.hc, self.J*s, check_tp=False)

    def __rmul__(self, other):
        # hopefully `other` is a scalar
        return self * other

    @classmethod
    def from_matrix(cls, m, spc_in, spc_out, espc_def=None, check_tp=True):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qudit('a', 2)
        >>> hb = qudit('b', 3)
        >>> hx = qudit('x', 5)
        >>> E1 = CP_Map.random(ha*hb, hx)
        >>> E2 = CP_Map.random(hx, ha*hb)
        >>> m = E2.as_matrix() * E1.as_matrix()
        >>> E3 = CP_Map.from_matrix(m, ha*hb, ha*hb)
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

        if field.mat_norm(np.transpose(np.conj(t)) - t) > toler:
            raise ValueError("matrix didn't correspond to a totally positive "+
                "superoperator (cross operator not self-adjoint)")

        (ew, ev) = field.mat_eig(t, True)

        if np.min(ew) < -toler:
            raise ValueError("matrix didn't correspond to a totally positive "+
                "superoperator (min eig="+str(np.min(ew))+")")
        ew = np.where(ew < 0, 0, ew)

        hc = cls._make_environ_spc(espc_def, ha.base_field, da*db)

        J = (hb * hc * ha.H).array()

        for i in range(da*db):
            J[{ hc: i }] = (hb * ha.H).array(ev[:,i] * np.sqrt(ew[i]), reshape=True)

        return cls(ha, hb, hc, J, check_tp=check_tp)

    @classmethod
    def from_kraus(cls, ops, espc_def=None):
        ops = list(ops)
        op_spc = ops[0].space
        dc = len(ops)
        hc = cls._make_environ_spc(espc_def, op_spc.base_field, dc)
        J = (op_spc * hc).array()
        for (i, op) in enumerate(ops):
            J[{ hc: i }] = op

        return cls(op_spc.bra_space(), op_spc.ket_space(), hc, J)

    def __add__(self, other):
        ret = super(CP_Map, self).__add__(other)
        if isinstance(other, CP_Map):
            return CP_Map.from_matrix(ret._m, ret.ha, ret.hb, check_tp=False)
        else:
            return ret

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
        >>> from qitensor.experimental.superop import CP_Map
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> U = ha.random_unitary()
        >>> rho = (ha*hb).random_density()
        >>> E = CP_Map.unitary(U)
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
        >>> from qitensor.experimental.superop import CP_Map
        >>> ha = qubit('a')
        >>> hb = qubit('b')
        >>> rho = (ha*hb).random_density()
        >>> E = CP_Map.identity(ha)
        >>> (E(rho) - rho).norm() < 1e-14
        True
        """

        return cls.unitary(spc.eye(), espc_def)

    @classmethod
    def totally_noisy(cls, spc, espc_def=None):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import CP_Map
        >>> ha = qudit('a', 5)
        >>> rho = ha.random_density()
        >>> E = CP_Map.totally_noisy(ha)
        >>> (E(rho) - ha.fully_mixed()).norm() < 1e-14
        True
        """

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
        >>> from qitensor.experimental.superop import CP_Map
        >>> ha = qudit('a', 5)
        >>> rho = ha.random_density()
        >>> E = CP_Map.noisy(ha, 0.2)
        >>> (E(rho) - 0.8*rho - 0.2*ha.fully_mixed()).norm() < 1e-14
        True
        """

        assert 0 <= p <= 1
        E0 = cls.totally_noisy(spc)
        E1 = cls.identity(spc)
        return p*E0 + (1-p)*E1

#ha = qudit('a', 2)
#hb = qudit('b', 2)
#hd = qudit('d', 3)
#rho = (ha*hd).random_density()
#E = CP_Map.random(ha, hb)
