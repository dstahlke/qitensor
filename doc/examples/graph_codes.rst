Sage: Graph Codes
=================

This is an implementation of the algorithm described in
"Quantum-error-correcting codes using qudit graph states",
Shiang Yong Looi, Li Yu, Vlad Gheorghiu, and Robert B. Griffiths,
Phys. Rev. A 78, 042303 (2008).

    >>> from qitensor import *
    >>> import numpy as np

    >>> #graph = graphs.CycleGraph(4)
    >>> #graph.allow_multiple_edges(True)
    >>> #graph.add_edge((0, 1))

    >>> graph = graphs.CubeGraph(3)
    >>> graph.relabel()

    >>> D = 2
    >>> n = graph.order()
    >>> (D, n)
    (2, 8)

    >>> graph.show() # long time

    >>> hilb = [ qudit('H_%d' % i, D, dtype=complex) for i in range(n) ]
    >>> c0 = np.product([h.x_plus() for h in hilb])
    >>> for (l, m, _label) in graph.edges():
    ...       c0 = cnot(hilb[l], hilb[m]) * c0
    ...
    >>> c0.space
    |H_0,H_1,H_2,H_3,H_4,H_5,H_6,H_7>

    >>> import itertools
    >>> nS = D^n
    >>> hS = qudit('S', nS)
    >>> cv_arr = (np.product(hilb) * hS.H).array()
    >>> for (i, a) in enumerate(itertools.product(range(D), repeat=n)):
    ...       cv_i = c0
    ...       for j in range(n):
    ...           cv_i = hilb[j].pauliZ(a[j]) * cv_i
    ...       cv_arr[{ hS.H: i }] = cv_i
    ...
    >>> cv_arr.space
    |H_0,H_1,H_2,H_3,H_4,H_5,H_6,H_7><S|

    >>> Q_size1 = lambda H: [ H.pauliX(i) * H.pauliZ(j) for i in range(D) \
    ...     for j in range(D) if (i>0 or j>0) ]
    >>> Q_list = []
    >>> for H in hilb:
    ...       Q_list += Q_size1(H)
    ...
    >>> len(Q_list)
    24

    >>> # Because of Eq. 17 this could be made a lot faster.  But for
    >>> # simplicity this is not taken advantage of.
    >>> cv_Q_list = [cv_arr.H * Q for Q in Q_list]
    >>> delta_matrix = np.zeros((nS, nS), dtype=int)
    >>> delta_matrix[:] = 3
    >>> for (i1, Q1) in enumerate(Q_list):
    ...       Q1_cv = Q1 * cv_arr
    ...       for (i2, cv_Q2) in enumerate(cv_Q_list):
    ...           if i1!=i2:
    ...               coeffs = cv_Q2 * Q1_cv
    ...               nonzero = abs(coeffs.nparray) > 1e-14
    ...               delta_matrix[nonzero] = 2
    ...
    >>> for cv_Q in cv_Q_list:
    ...       coeffs = cv_Q * cv_arr
    ...       nonzero = abs(coeffs.nparray) > 1e-14
    ...       delta_matrix[nonzero] = 1
    ...

    >>> delta_matrix.shape
    (256, 256)

    >>> delta_matrix
    array([[3, 1, 1, ..., 2, 2, 2],
           [1, 3, 2, ..., 2, 2, 2],
           [1, 2, 3, ..., 2, 2, 2],
           ..., 
           [2, 2, 2, ..., 3, 2, 1],
           [2, 2, 2, ..., 2, 3, 1],
           [2, 2, 2, ..., 1, 1, 3]])

    >>> S_matrix = np.where(delta_matrix >= 2, 1, 0)
    >>> S_graph = Graph(S_matrix)
    >>> len(S_graph.clique_maximum())
    64

    >>> S_matrix = np.where(delta_matrix >= 3, 1, 0)
    >>> S_graph = Graph(S_matrix)
    >>> len(S_graph.clique_maximum())
    8
