import numpy as np
import itertools

from qitensor import qudit, direct_sum, NotKetSpaceError, \
    HilbertSpace, HilbertArray, HilbertError
from qitensor.space import create_space2

toler = 1e-12

# FIXME - use exceptions rather than assert
# FIXME - some methods don't have docs
# FIXME - use CP_Map in the map-state duality example

__all__ = ['Superoperator', 'CP_Map']

def _unreduce_supop_v1(in_space, out_space, m):
    """
    This is the function that handles restoring a pickle.
    """
    return Superoperator(in_space, out_space, m)

class Superoperator(object):
    def __init__(self, in_space, out_space, m):
        self._in_space = self._to_ket_space(in_space)
        self._out_space = self._to_ket_space(out_space)
        self._m = np.matrix(m)

        assert m.shape == (self.out_space.O.dim(), self.in_space.O.dim())

    def __reduce__(self):
        """
        Tells pickle how to store this object.

        >>> import pickle
        >>> from qitensor import qubit, qudit
        >>> from qitensor.experimental.superop import Superoperator, CP_Map
        >>> ha = qudit('a', 3)
        >>> hb = qubit('b')
        >>> rho = (ha*hb).random_density()

        >>> E = Superoperator.from_function(ha, lambda x: x.T)
        >>> E
        Superoperator( |a><a| to |a><a| )
        >>> F = pickle.loads(pickle.dumps(E))
        >>> F
        Superoperator( |a><a| to |a><a| )
        >>> (E(rho) - F(rho)).norm > 1e-14
        True
        """

        return _unreduce_supop_v1, (self.in_space, self.out_space, self._m)

    @property
    def in_space(self):
        return self._in_space

    @property
    def out_space(self):
        return self._out_space

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
        return 'Superoperator( '+str(self.in_space.O)+' to '+str(self.out_space.O)+' )'

    def __repr__(self):
        return str(self)

    def as_matrix(self):
        return self._m

    def __call__(self, rho):
        assert rho.space.bra_ket_set >= self.in_space.O.bra_ket_set
        (row_space, col_space) = rho._get_row_col_spaces(col_space=self.in_space.O)
        ret_vec = self._m * rho.as_np_matrix(col_space=self.in_space.O)
        if len(row_space):
            out_space = self.out_space.O * np.prod(row_space)
        else:
            out_space = self.out_space.O
        return out_space.array(ret_vec, reshape=True, input_axes=self.out_space.O.axes+row_space)

    def __mul__(self, other):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qudit('a', 2)
        >>> hb = qudit('b', 3)
        >>> hc = qudit('c', 4)
        >>> hd = qudit('d', 2)
        >>> he = qudit('e', 3)

        >>> rho = (ha*hd).O.random_array()
        >>> E = Superoperator.random(ha, ha)
        >>> E
        Superoperator( |a><a| to |a><a| )

        >>> 2*E
        Superoperator( |a><a| to |a><a| )
        >>> ((2*E)(rho) - 2*E(rho)).norm() < 1e-14
        True

        >>> (-2)*E
        Superoperator( |a><a| to |a><a| )
        >>> (((-2)*E)(rho) - (-2)*E(rho)).norm() < 1e-14
        True

        >>> E1 = Superoperator.random(ha, hb*hc, 'env1')
        >>> E1
        Superoperator( |a><a| to |b,c><b,c| )
        >>> E2 = Superoperator.random(hc*hd, he, 'env2')
        >>> E2
        Superoperator( |c,d><c,d| to |e><e| )
        >>> E3 = E2*E1
        >>> E3
        Superoperator( |a,d><a,d| to |b,e><b,e| )
        >>> (E2(E1(rho)) - E3(rho)).norm() < 1e-12 # FIXME - why not 1e-14 precision?
        True
        """

        if isinstance(other, Superoperator):
            common_spc = self.in_space.ket_set & other.out_space.ket_set
            in_spc = (self.in_space.ket_set - common_spc) | other.in_space.ket_set
            in_spc = create_space2(in_spc , frozenset())
            return Superoperator.from_function(in_spc, lambda x: self(other(x)))

        if isinstance(other, HilbertArray):
            return NotImplemented

        # hopefully `other` is a scalar
        return Superoperator(self.in_space, self.out_space, self._m*other)

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
            return NotImplemented

        assert self.in_space == other.in_space
        assert self.out_space == other.out_space

        return Superoperator(self.in_space, self.out_space, self._m + other._m)

    def __neg__(self):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qudit('a', 4)
        >>> hb = qudit('b', 3)
        >>> E = CP_Map.random(ha, hb)
        >>> rho = ha.random_density()
        >>> ((-E)(rho) + E(rho)).norm() < 1e-14
        True
        """

        return (-1)*self

    def __sub__(self, other):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import Superoperator
        >>> ha = qudit('a', 4)
        >>> hb = qudit('b', 3)
        >>> E1 = CP_Map.random(ha, hb)
        >>> E2 = CP_Map.random(ha, hb)
        >>> rho = ha.random_density()
        >>> chi = (E1 - E2)(rho)
        >>> xi  = E1(rho) - E2(rho)
        >>> (chi - xi).norm() < 1e-14
        True
        """

        return self + (-other)

    @classmethod
    def from_function(cls, in_space, f):
        """
        >>> from qitensor import qubit, qudit
        >>> from qitensor.experimental.superop import Superoperator, CP_Map
        >>> ha = qudit('a', 3)
        >>> hb = qubit('b')
        >>> rho = (ha*hb).random_density()

        >>> ET = Superoperator.from_function(ha, lambda x: x.T)
        >>> ET
        Superoperator( |a><a| to |a><a| )
        >>> (ET(rho) - rho.transpose(ha)).norm() < 1e-14
        True

        >>> Superoperator.from_function(ha, lambda x: x.H)
        Traceback (most recent call last):
            ...
        ValueError: function was not linear

        >>> CP_Map.from_function(ha, lambda x: x.T)
        Traceback (most recent call last):
            ...
        ValueError: matrix didn't correspond to a completely positive superoperator (min eig=-1.0)

        >>> U = ha.random_unitary()
        >>> EU = CP_Map.from_function(ha, lambda x: U*x*U.H)
        >>> EU
        CP_Map( |a><a| to |a><a| )
        >>> (EU(rho) - U*rho*U.H).norm() < 1e-14
        True
        """

        in_space = cls._to_ket_space(in_space)
        out_space = f(in_space.eye()).space
        assert out_space == out_space.H
        out_space = out_space.ket_space()

        m = np.zeros((out_space.dim()**2, in_space.dim()**2), in_space.base_field.dtype)
        for (i, x) in enumerate(in_space.O.index_iter()):
            m[:, i] = f(in_space.O.basis_vec(x)).nparray.flatten()

        E = Superoperator(in_space, out_space, m)

        rho = in_space.random_density()
        if (E(rho) - f(rho)).norm() > toler:
            raise ValueError('function was not linear')

        if cls == CP_Map:
            E = E.upgrade_to_cp_map()

        return E

    @classmethod
    def random(cls, spc_in, spc_out, espc_def=None):
        in_space = cls._to_ket_space(spc_in)
        out_space = cls._to_ket_space(spc_out)
        m = spc_in.base_field.random_array((out_space.O.dim(), in_space.O.dim()))
        return Superoperator(in_space, out_space, m)

    def upgrade_to_cp_map(self, espc_def=None):
        return CP_Map.from_matrix(self._m, self.in_space, self.out_space, espc_def=espc_def)

    def upgrade_to_cptp_map(self, espc_def=None):
        ret = self.upgrade_to_cp_map()
        ret.assert_cptp()
        return ret

def _unreduce_cpmap_v1(in_space, out_space, env_space, J):
    """
    This is the function that handles restoring a pickle.
    """
    return CP_Map(in_space, out_space, env_space, J)

class CP_Map(Superoperator):
    # FIXME - only env_space is really needed
    def __init__(self, in_space, out_space, env_space, J, _complimentary_channel=None):
        """
        >>> ha = qudit('a', 2)
        >>> hb = qudit('b', 2)
        >>> hd = qudit('d', 3)
        >>> rho = (ha*hd).random_density()
        >>> E = CP_Map.random(ha, hb)
        >>> ((E.J * rho * E.J.H).trace(E.env_space) - E(rho)).norm() < 1e-14
        True
        """

        in_space = self._to_ket_space(in_space)
        out_space = self._to_ket_space(out_space)
        env_space = self._to_ket_space(env_space)

        assert J.space == out_space * env_space * in_space.H

        da = in_space.dim()
        db = out_space.dim()
        t = np.zeros((db, da, db, da), dtype=in_space.base_field.dtype)
        for j in env_space.index_iter():
            op = J[{ env_space: j }].as_np_matrix(row_space=in_space.H)
            t += np.tensordot(op, op.conj(), axes=([],[]))
        t = t.transpose([0,2,1,3])
        t = t.reshape(db**2, da**2)
        t = np.matrix(t)

        super(CP_Map, self).__init__(in_space, out_space, t)

        self._J = J
        self._env_space = env_space

        if _complimentary_channel is None:
            self._C = CP_Map(self.in_space, self.env_space, self.out_space, self.J, \
                _complimentary_channel=self)
        else:
            self._C = _complimentary_channel

    def __reduce__(self):
        """
        Tells pickle how to store this object.

        >>> import pickle
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import CP_Map
        >>> ha = qudit('a', 2)
        >>> rho = ha.O.random_array()
        >>> E = CP_Map.random(ha, ha)
        >>> F = pickle.loads(pickle.dumps(E))
        >>> F
        CP_Map( |a><a| to |a><a| )
        >>> (E(rho) - F(rho)).norm > 1e-14
        True
        """

        return _unreduce_cpmap_v1, (self.in_space, self.out_space, self.env_space, self.J)

    @property
    def env_space(self):
        return self._env_space

    @property
    def J(self):
        """The channel isometry."""
        return self._J

    @property
    def C(self):
        """The complimentary channel."""
        return self._C

    def is_cptp(self):
        return (self.J.H*self.J - self.in_space.eye()).norm() < toler

    def assert_cptp(self):
        if not self.is_cptp():
            raise ValueError('channel is not trace preserving')

    def __str__(self):
        return 'CP_Map( '+str(self.in_space.O)+' to '+str(self.out_space.O)+' )'

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import CP_Map
        >>> ha = qudit('a', 2)
        >>> hb = qudit('b', 3)
        >>> hc = qudit('c', 2)
        >>> hd = qudit('d', 2)
        >>> he = qudit('e', 3)

        >>> rho = (ha*hd).O.random_array()
        >>> E = CP_Map.random(ha, ha)
        >>> E
        CP_Map( |a><a| to |a><a| )

        >>> 2*E
        CP_Map( |a><a| to |a><a| )
        >>> ((2*E)(rho) - 2*E(rho)).norm() < 1e-14
        True

        >>> (-2)*E
        Superoperator( |a><a| to |a><a| )
        >>> (((-2)*E)(rho) - (-2)*E(rho)).norm() < 1e-14
        True

        >>> E1 = CP_Map.random(ha, hb*hc, 'env1')
        >>> E1
        CP_Map( |a><a| to |b,c><b,c| )
        >>> E2 = CP_Map.random(hc*hd, he, 'env2')
        >>> E2
        CP_Map( |c,d><c,d| to |e><e| )
        >>> E3 = E2*E1
        >>> E3
        CP_Map( |a,d><a,d| to |b,e><b,e| )
        >>> E3.env_space
        |env1,env2>
        >>> (E2(E1(rho)) - E3(rho)).norm() < 1e-14
        True
        """

        if isinstance(other, Superoperator):
            common_spc = self.in_space.ket_set & other.out_space.ket_set
            in_spc  = (self.in_space.ket_set - common_spc) | other.in_space.ket_set
            out_spc = self.out_space.ket_set | (other.out_space.ket_set - common_spc)
            # FIXME - will break in the case of E*E
            env     = self.env_space * other.env_space
            in_spc  = create_space2(in_spc , frozenset())
            out_spc = create_space2(out_spc, frozenset())
            return CP_Map(in_spc, out_spc, env, self.J*other.J)

        if isinstance(other, HilbertArray):
            return NotImplemented

        # hopefully `other` is a scalar
        if other < 0:
            return Superoperator.__mul__(self, other)
        else:
            s = self.in_space.base_field.sqrt(other)
            return CP_Map(self.in_space, self.out_space, self.env_space, self.J*s)

    def __rmul__(self, other):
        # hopefully `other` is a scalar
        return self * other

    def __add__(self, other):
        ret = super(CP_Map, self).__add__(other)
        if isinstance(other, CP_Map):
            return ret.upgrade_to_cp_map()
        else:
            return ret

    def add2(self, other):
        """
        >>> import numpy.linalg as linalg
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import CP_Map
        >>> ha = qudit('a', 2)
        >>> hb = qudit('b', 3)
        >>> E1 = CP_Map.random(ha, hb, 'hc1')
        >>> E2 = CP_Map.random(ha, hb, 'hc2')
        >>> X = E1 + E2
        >>> Y = E1.add2(E2)
        >>> linalg.norm(X.as_matrix() - Y.as_matrix()) < 1e-14
        True
        >>> (E1.env_space, E2.env_space, Y.env_space)
        (|hc1>, |hc2>, |hc1+hc2>)
        """
        # FIXME - docs

        if not isinstance(other, CP_Map):
            raise ValueError('other was not a CP_Map')

        assert self.in_space == other.in_space
        assert self.out_space == other.out_space
        ret_hc = direct_sum((self.env_space, other.env_space))
        ret_J = ret_hc.P[0]*self.J + ret_hc.P[1]*other.J
        return CP_Map(self.in_space, self.out_space, ret_hc, ret_J)

    @classmethod
    def from_matrix(cls, m, spc_in, spc_out, espc_def=None):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import CP_Map
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

        in_space = cls._to_ket_space(spc_in)
        out_space = cls._to_ket_space(spc_out)
        da = in_space.dim()
        db = out_space.dim()
        t = np.array(m)
        assert t.shape == (db*db, da*da)
        t = t.reshape(db, db, da, da)
        t = t.transpose([0, 2, 1, 3])
        t = t.reshape(db*da, db*da)

        field = in_space.base_field

        if field.mat_norm(np.transpose(np.conj(t)) - t) > toler:
            raise ValueError("matrix didn't correspond to a completely positive "+
                "superoperator (cross operator not self-adjoint)")

        (ew, ev) = field.mat_eig(t, True)

        if np.min(ew) < -toler:
            raise ValueError("matrix didn't correspond to a completely positive "+
                "superoperator (min eig="+str(np.min(ew))+")")
        ew = np.where(ew < 0, 0, ew)

        env_space = cls._make_environ_spc(espc_def, in_space.base_field, da*db)

        J = (out_space * env_space * in_space.H).array()

        for i in range(da*db):
            J[{ env_space: i }] = (out_space * in_space.H).array(ev[:,i] * field.sqrt(ew[i]), reshape=True)

        return CP_Map(in_space, out_space, env_space, J)

    @classmethod
    def from_kraus(cls, ops, espc_def=None):
        ops = list(ops)
        op_spc = ops[0].space
        dc = len(ops)
        env_space = cls._make_environ_spc(espc_def, op_spc.base_field, dc)
        J = (op_spc * env_space).array()
        for (i, op) in enumerate(ops):
            J[{ env_space: i }] = op

        return CP_Map(op_spc.bra_space(), op_spc.ket_space(), env_space, J)

    @classmethod
    def random(cls, spc_in, spc_out, espc_def=None):
        in_space = cls._to_ket_space(spc_in)
        out_space = cls._to_ket_space(spc_out)
        dc = in_space.dim() * out_space.dim()
        env_space = cls._make_environ_spc(espc_def, in_space.base_field, dc)
        J = (out_space*env_space*in_space.H).random_isometry()
        return CP_Map(in_space, out_space, env_space, J)

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

        in_space = U.space.bra_space().H
        out_space = U.space.ket_space()
        env_space = cls._make_environ_spc(espc_def, in_space.base_field, 1)
        J = U * env_space.ket(0)
        return CP_Map(in_space, out_space, env_space, J)

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

        in_space = cls._to_ket_space(spc)
        d = in_space.dim()
        d2 = d*d
        env_space = cls._make_environ_spc(espc_def, in_space.base_field, d2)
        J = (in_space.O*env_space).array()
        for (i, (j, k)) in enumerate(itertools.product(in_space.index_iter(), repeat=2)):
            J[{ in_space.H: j, in_space: k, env_space: i }] = 1
        J /= in_space.base_field.sqrt(d)
        return CP_Map(in_space, in_space, env_space, J)

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

    @classmethod
    def decohere(cls, spc, espc_def=None):
        """
        >>> from qitensor import qudit
        >>> from qitensor.experimental.superop import CP_Map
        >>> ha = qudit('a', 5)
        >>> rho = ha.random_density()
        >>> E = CP_Map.decohere(ha)
        >>> (E(rho) - ha.diag(rho.diag(as_np=True))).norm() < 1e-14
        True
        """

        in_space = cls._to_ket_space(spc)
        d = in_space.dim()
        env_space = cls._make_environ_spc(espc_def, in_space.base_field, d)
        J = (in_space.O*env_space).array()
        for (i, a) in enumerate(in_space.index_iter()):
            J[{ in_space.H: a, in_space: a, env_space: i }] = 1
        return CP_Map(in_space, in_space, env_space, J)
