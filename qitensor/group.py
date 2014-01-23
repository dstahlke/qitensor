"""
Simple implementation of groups, based upon Cayley tables.
"""

__all__ = ['Group', 'dihedral_group']

def _unreduce_group_element_v1(group, idx):
    return group.elements[idx]

class GroupElement(object):
    def __init__(self, group, idx, name):
        self.group = group
        self.idx = idx
        self.name = name

    def __reduce__(self):
        """
        Tells pickle how to store this object.
        """
        return _unreduce_group_element_v1, (self.group, self.idx)

    def __hash__(self): return self.idx

    def __repr__(self):
        return "<"+str(self.group)+"."+str(self)+">"

    def __str__(self): return self.name

    def __cmp__(self, other):
        assert self.group == other.group
        return cmp(self.idx, other.idx)

    def __mul__(self, other):
        return self.group.mtab[self.idx][other.idx]

    def __div__(self, other):
        return self * other.inv()

    def __truediv__(self, other):
        return self * other.inv()

    def inv(self): return self.group.itab[self.idx]

class Group(object):
    def __init__(self, group_name, element_names, m_table):
        self.name = group_name
        self.elements = [GroupElement(self, i, name) for (i, name) in enumerate(element_names)]
        self.order = len(self.elements)
        self.e = self.elements[0]
        element_by_name = { name: self.elements[i] for (i, name) in enumerate(element_names) }
        self.mtab = [[ element_by_name[name] for name in row ] for row in m_table ]

        self.itab = []
        for f in self.elements:
            h = None
            for g in self.elements:
                if f*g == self.e:
                    assert h is None
                    h = g
            assert h is not None
            self.itab.append(h)

        self._assert_is_group()

    def _assert_is_group(self):
        for f in self.elements:
            assert f/f == self.e
            for g in self.elements:
                assert (f/g)*g == f
                for h in self.elements:
                    assert (f*g)*h == f*(g*h)
        for f in self.elements:
            assert f.inv() * f == self.e
            assert f * f.inv() == self.e

    def __repr__(self):
        return '<Group('+self.name+')>'

    def __str__(self): return self.name

class DihedralGroup_impl(Group):
    def __init__(self, n=0, trust_me=False):
        if not trust_me:
            raise Exception('construct this through the dihedral_group(n) function')

        r = ['r%d' % i for i in range(n)]
        s = ['s%d' % i for i in range(n)]
        elements = r + s

        mtab = [[None for i in range(n*2)] for j in range(n*2)]
        for i in range(n):
            for j in range(n):
                mtab[i  ][j  ] = r[(i+j)%n]
                mtab[i  ][j+n] = s[(i+j)%n]
                mtab[i+n][j  ] = s[(i-j)%n]
                mtab[i+n][j+n] = r[(i-j)%n]

        Group.__init__(self, 'S%d'%n, elements, mtab)
        self.r = self.elements[:n]
        self.s = self.elements[n:]

        for i in range(n):
            self.__dict__[r[i]] = self.r[i]
            self.__dict__[s[i]] = self.s[i]

    def __reduce__(self):
        """
        Tells pickle how to store this object.
        """
        return _unreduce_dihedral_group_v1, (len(self.r),)

_dihedral_group_cache = {}
def dihedral_group(n):
    """
    Return the dihedral group S_n.

    >>> from qitensor import dihedral_group
    >>> S3 = dihedral_group(3)
    >>> S3.order
    6
    >>> S3.elements
    [<S3.r0>, <S3.r1>, <S3.r2>, <S3.s0>, <S3.s1>, <S3.s2>]
    >>> S3.e
    <S3.r0>
    >>> S3.r1 * S3.s0
    <S3.s1>

    >>> import pickle
    >>> id(S3) == id(pickle.loads(pickle.dumps(S3)))
    True
    """

    if n not in _dihedral_group_cache:
        _dihedral_group_cache[n] = DihedralGroup_impl(n, True)
    return _dihedral_group_cache[n]

_unreduce_dihedral_group_v1 = dihedral_group
