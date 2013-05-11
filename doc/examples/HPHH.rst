Creation of channel from a function
===================================

This example creates a quantum channel whose action is given by a function `f(x)`.  It then
shows that the channel has private capacity.  The channel definition is from
*Low-Dimensional Bound Entanglement With One-Way Distillable Cryptographic Key* by
Karol Horodecki, Łukasz Pankowski, Michał Horodecki, and Paweł Horodecki.
I didn't read the paper, so this is somewhat based upon hearsay.

    >>> import numpy as np
    >>> from qitensor import qubit, CP_Map

    >>> hA = qubit('A')
    >>> hB = qubit('B')
    >>> hx = qubit('X')

    >>> s = np.sqrt(2) / (8*(1 + np.sqrt(2)))
    >>> t = 1 / (4*(1 + np.sqrt(2)))
    >>> rho_H = (hA*hA.prime*hB*hB.prime).O.array([
    ...     [ s, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  s, 0, 0, 0],
    ...     [ 0, s, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, s, 0],
    ...     [ 0, 0, s, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, s, 0, 0],
    ...     [ 0, 0, 0, s,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0,-s],
    ...
    ...     [ 0, 0, 0, 0,  t, 0, 0, 0,  s, 0, 0, s,  0, 0, 0, 0],
    ...     [ 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
    ...     [ 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
    ...     [ 0, 0, 0, 0,  0, 0, 0, t,  s, 0, 0,-s,  0, 0, 0, 0],
    ...
    ...     [ 0, 0, 0, 0,  s, 0, 0, s,  t, 0, 0, 0,  0, 0, 0, 0],
    ...     [ 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
    ...     [ 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0],
    ...     [ 0, 0, 0, 0,  s, 0, 0,-s,  0, 0, 0, t,  0, 0, 0, 0],
    ...
    ...     [ s, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  s, 0, 0, 0],
    ...     [ 0, 0, s, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, s, 0, 0],
    ...     [ 0, s, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, s, 0],
    ...     [ 0, 0, 0,-s,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, s],
    ... ], reshape=True, input_axes=(hA, hB, hA.prime, hB.prime, hA.H, hB.H, hA.prime.H, hB.prime.H))

    >>> # Check that it is PT (partial transpose) invariant
    >>> (rho_H - rho_H.transpose(hA*hA.prime)).norm() < 1e-14
    True

    >>> # Make a superoperator whose action is given by the function f(x).  A check is automatically
    >>> # made to ensure that f(x) is a completely positive map.
    >>> def f(x):
    ...     return 4*(x.T * rho_H).trace(hA*hA.prime)
    >>> N = CP_Map.from_function(hA*hA.prime, f, espc_def='E')
    >>> N
    CP_Map( |A,A'><A,A'| to |B,B'><B,B'| )

    >>> # Show that the channel has private capacity
    >>> rho_A_0 = hA.ket(0).O * hA.prime.fully_mixed()
    >>> rho_A_1 = hA.ket(1).O * hA.prime.fully_mixed()
    >>> ensemble = [0.5*rho_A_0, 0.5*rho_A_1]
    >>> "%.6f" % N.private_information(ensemble)
    '0.021340'

    >>> # N.J gives the channel isometry.
    >>> N.J.space
    |B,B',E><A,A'|
    >>> # The channel can act on a state like so:
    >>> sigma = N( (hA*hA.prime).random_density() )
    >>> sigma.space
    |B,B'><B,B'|
