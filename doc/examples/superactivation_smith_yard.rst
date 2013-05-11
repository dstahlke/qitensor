Superactivation of channels by Smith + Yard
===========================================

Play with the protocol given in supplementary material of *Quantum communication with
zero-capacity channels* by Smith + Yard.
This demonstrates creation of a channel from Kraus operators.

    >>> import numpy as np
    >>> from qitensor import qubit, qudit, max_entangled, CP_Map, Superoperator

    >>> ha1 = qubit('a1')   # First  qubit of channel input
    >>> ha2 = qubit('a2')   # Second qubit of channel input
    >>> hb1 = qubit('b1')   # First  qubit of channel output
    >>> hb2 = qubit('b2')   # Second qubit of channel output
    >>> he  = qudit('e', 6) # Environment

    >>> # Make the six Kraus operators.
    >>> q = np.sqrt(2) / (1 + np.sqrt(2))
    >>> M0 = ha2.diag([np.sqrt(2+np.sqrt(2))/2, np.sqrt(2-np.sqrt(2))/2])
    >>> # Note: Smith and Yard are missing the minus sign in their paper.
    >>> M1 = ha2.diag([np.sqrt(2-np.sqrt(2))/2, -np.sqrt(2+np.sqrt(2))/2])
    >>> Klist = []
    >>> Klist.append(np.sqrt(q/2) * ha1.eye() * ha2.ket(0).O)
    >>> Klist.append(np.sqrt(q/2) * ha1.Z     * ha2.ket(1).O)
    >>> Klist.append(np.sqrt(q/4) * ha1.Z     * ha2.Y)
    >>> Klist.append(np.sqrt(q/4) * ha1.eye() * ha2.X)
    >>> Klist.append(np.sqrt(1-q) * ha1.X     * M0)
    >>> Klist.append(np.sqrt(1-q) * ha1.Y     * M1)

    >>> # Relabel the ket spaces of the Krauss operators so that the channel output is on b1,b2.
    >>> # The channel maps operators on a1,a2 to operators on b1,b2.
    >>> Klist = [K.relabel({ ha1: hb1, ha2: hb2 }) for K in Klist]

    >>> # Make the channel.
    >>> N = CP_Map.from_kraus(Klist, he)
    >>> # Show what spaces the channel isometry acts on.
    >>> N.J.space
    |b1,b2,e><a1,a2|

    >>> # Check that it is PPT.  The channel `N` is concatenated with the transposer channel
    >>> # (which is not completely positive, but is a superoperator).  The result is a
    >>> # superoperator.  The `upgrade_to_cptp_map` method turns a Superoperator into a CP_Map,
    >>> # raising an exception if the map is not completely positive.  The fact that no error
    >>> # is raised here means that `N` has positive partial transpose.
    >>> (N * Superoperator.transposer(N.in_space)).upgrade_to_cptp_map()
    CP_Map( |a1,a2><a1,a2| to |b1,b2><b1,b2| )

    >>> # Check that it has positive private information.
    >>> rho0 = ha1.ket(0).O * ha2.fully_mixed()
    >>> rho1 = ha1.ket(1).O * ha2.fully_mixed()
    >>> ensemble = [0.5*rho0, 0.5*rho1]
    >>> "%.6f" % N.private_information(ensemble)
    '0.021340'

Now show that the protocol given in the paper leads to positive coherent information,
therefore a positive capacity.

    >>> hx = qudit('x', len(ensemble))
    >>> rho_ax = np.sum([ hx.ket(i).O * rho_i for (i, rho_i) in enumerate(ensemble) ])
    >>> rho_ax.space
    |a1,a2,x><a1,a2,x|

    >>> # Create the 50% erasure channel.
    >>> hc1 = qubit('c1')
    >>> hc2 = qubit('c2')
    >>> # The channel output is on an automatically created space labeled `'d'`.  The
    >>> # environment (output of the complimentary channel) is `'f'`.
    >>> A = CP_Map.erasure(hc1*hc2, 0.5, 'd', 'f')
    >>> A
    CP_Map( |c1,c2><c1,c2| to |d><d| )
    >>> A.C # complimentary channel
    CP_Map( |c1,c2><c1,c2| to |f><f| )
    >>> hd = A.out_space
    >>> hf = A.env_space
    >>> (hd, hf)
    (|d>, |f>)

    >>> # Compose the two channels.  Since they act on different spaces (the input of `N` is
    >>> # disjoint from the output of `A`), the channels are put in parallel.
    >>> NA = N * A
    >>> NA
    CP_Map( |a1,a2,c1,c2><a1,a2,c1,c2| to |b1,b2,d><b1,b2,d| )

    >>> # Do the special purification
    >>> psi = (hc1*ha1*hx).array()
    >>> psi[0,0,0] = 1/np.sqrt(2)
    >>> psi[1,1,1] = 1/np.sqrt(2)
    >>> psi *= max_entangled(hc2, ha2)
    >>> psi.space
    |a1,a2,c1,c2,x>
    >>> (psi.O.trace(hc1*hc2) - rho_ax).norm() < 1e-12
    True
    >>> rho_ac = psi.O.trace(hx)

    >>> # Coherent information for N*A is half the private information of N
    >>> "%.6f" % NA.coherent_information(rho_ac)
    '0.010670'
