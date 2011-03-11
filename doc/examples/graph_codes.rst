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

    >>> # Create the graph state
    >>> hilb = [ qudit('H_%d' % i, D, dtype=complex) for i in range(n) ]
    >>> c0 = np.product([h.x_plus() for h in hilb])
    >>> for (l, m, _label) in graph.edges():
    ...       c0 = cnot(hilb[l], hilb[m]) * c0
    ...
    >>> c0.space
    |H_0,H_1,H_2,H_3,H_4,H_5,H_6,H_7>

    >>> # Create the graph basis
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

    >>> # Create the size-1 Pauli operators
    >>> Q_list = []
    >>> for H in hilb:
    ...     for i in range(D):
    ...         for j in range(D):
    ...             if i>0 or j>0:
    ...                 Q_list.append(H.pauliX(i) * H.pauliZ(j))
    ...
    >>> len(Q_list)
    24

    >>> # Compute the Pauli distance between graph basis states.
    >>> # Because of Eq. 17 this could be made a lot faster.  But for
    >>> # simplicity this is not taken advantage of.
    >>> cv_Q_list = [cv_arr.H * Q for Q in Q_list]
    >>> delta_matrix = np.zeros((nS, nS), dtype=int)
    >>> delta_matrix[:] = 3
    >>> for (i1, Q1) in enumerate(Q_list):
    ...     Q1_cv = Q1 * cv_arr
    ...     for (i2, cv_Q2) in enumerate(cv_Q_list):
    ...         if i1 != i2:
    ...             coeffs = cv_Q2 * Q1_cv
    ...             nonzero = abs(coeffs.nparray) > 1e-14
    ...             delta_matrix[nonzero] = 2
    ...
    >>> for cv_Q in cv_Q_list:
    ...     coeffs = cv_Q * cv_arr
    ...     nonzero = abs(coeffs.nparray) > 1e-14
    ...     delta_matrix[nonzero] = 1
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

    >>> # Create the S graph for delta=2, and find the maximum
    >>> # clique size.
    >>> S_matrix = np.where(delta_matrix >= 2, 1, 0)
    >>> S_graph = Graph(S_matrix)
    >>> len(S_graph.clique_maximum())
    64

    >>> # Create the S graph for delta=3, and find the maximum
    >>> # clique size.
    >>> S_matrix = np.where(delta_matrix >= 3, 1, 0)
    >>> S_graph = Graph(S_matrix)
    >>> len(S_graph.clique_maximum())
    8
