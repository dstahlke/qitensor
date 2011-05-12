Sage: Vectors Over the Symbolic Ring
====================================

    >>> from sage.all import * # doctest: +SKIP
    >>> from qitensor import qubit # doctest: +SKIP
    >>> ha = qubit('a', dtype=SR) # doctest: +SKIP
    >>> hb = qubit('b', dtype=SR) # doctest: +SKIP
    >>> (x, y) = var('x y') # doctest: +SKIP
    >>> s = ha.array([1, x]) # doctest: +SKIP
    >>> t = hb.array([1, y]) # doctest: +SKIP
    >>> s*t # doctest: +SKIP
    |a,b>
    [  1]
    [  y]
    [---]
    [  x]
    [x*y]

    >>> # density operator
    >>> (s*t).O # doctest: +SKIP
    |a,b><a,b|
    [                            1                  conjugate(y)|                 conjugate(x)     conjugate(x)*conjugate(y)]
    [                            y                y*conjugate(y)|               y*conjugate(x)   y*conjugate(x)*conjugate(y)]
    [-----------------------------------------------------------+-----------------------------------------------------------]
    [                            x                x*conjugate(y)|               x*conjugate(x)   x*conjugate(x)*conjugate(y)]
    [                          x*y              x*y*conjugate(y)|             x*y*conjugate(x) x*y*conjugate(x)*conjugate(y)]
    >>> # density operator with ``ha`` traced out
    >>> (s*t).O.trace(ha) # doctest: +SKIP
    |b><b|
    [                            x*conjugate(x) + 1     x*conjugate(x)*conjugate(y) + conjugate(y)]
    [                          x*y*conjugate(x) + y x*y*conjugate(x)*conjugate(y) + y*conjugate(y)]
    >>> # there is no entanglement for this separable state
    >>> (s*t).O.trace(ha).entropy(normalize=True) # doctest: +SKIP
    0
    >>> # this however is an entangled state
    >>> (ha*hb).array([[1/sqrt(5),0],[0,2/sqrt(5)]]).O.trace(ha).entropy() # doctest: +SKIP
    -1/5*log(1/5)/log2 - 4/5*log(4/5)/log2
    >>> # The entropy from a symbolic expression is a big mess.  And in this
    >>> # case it appears to not be correct since the resulting value is
    >>> # a complex number.
    >>> (ha*hb).array([[1,1],[1,exp(I*pi*x)]]).normalized().O.trace(ha).entropy(checks=False).simplify_full() # doctest: +SKIP
    sqrt(e^(2*I*pi*x) + 3)*sqrt(3*e^(2*I*pi*x) + 1)*((-2*I*pi - 2*I*pi*x)*e^(I*pi*x) + 2*(e^(1/2*I*pi*x) - 2*e^(I*pi*x) + e^(3/2*I*pi*x))*log(e^(1/2*I*pi*x) - 1) - 2*(e^(1/2*I*pi*x) + 2*e^(I*pi*x) + e^(3/2*I*pi*x))*log(e^(1/2*I*pi*x) + 1) + I*pi*e^(1/2*I*pi*x) + I*pi*e^(3/2*I*pi*x) + 2*e^(I*pi*x)*log(e^(2*I*pi*x) + 3) + 2*e^(I*pi*x)*log(3*e^(2*I*pi*x) + 1))/(10*e^(2*I*pi*x)*log(2) + 3*e^(4*I*pi*x)*log(2) + 3*log(2))


    >>> U = (ha * hb).eye() # doctest: +SKIP
    >>> # arrays can be indexed using dictionaries
    >>> U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 0 }] = x # doctest: +SKIP
    >>> U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 1 }] = y # doctest: +SKIP
    >>> U # doctest: +SKIP
    |a,b><a,b|
    [x y|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1]

    >>> U.I # doctest: +SKIP
    |a,b><a,b|
    [ 1/x -y/x|   0    0]
    [   0    1|   0    0]
    [---------+---------]
    [   0    0|   1    0]
    [   0    0|   0    1]

    >>> U * U.I # doctest: +SKIP
    |a,b><a,b|
    [1 0|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1]

    >>> ((U ** 3) * (U ** -3)).apply_map(lambda x: x.simplify_full()) # doctest: +SKIP
    |a,b><a,b|
    [1 0|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1]

    >>> M = ha.O.array([[1,1],[1,x]]) # doctest: +SKIP
    >>> (W, V) = M.eig() # doctest: +SKIP
    >>> W # doctest: +SKIP
    |a><a|
    [1/2*x + 1/2*sqrt(x^2 - 2*x + 5) + 1/2                                     0]
    [                                    0 1/2*x - 1/2*sqrt(x^2 - 2*x + 5) + 1/2]
    >>> V # doctest: +SKIP
    |a><a|
    [                                1/sqrt(1/4*(x + sqrt(x^2 - 2*x + 5) - 1)^2 + 1)                                 1/sqrt(1/4*(x - sqrt(x^2 - 2*x + 5) - 1)^2 + 1)]
    [1/2*(x + sqrt(x^2 - 2*x + 5) - 1)/sqrt(1/4*(x + sqrt(x^2 - 2*x + 5) - 1)^2 + 1) 1/2*(x - sqrt(x^2 - 2*x + 5) - 1)/sqrt(1/4*(x - sqrt(x^2 - 2*x + 5) - 1)^2 + 1)]
    >>> (V.H * M * V - W).apply_map(lambda s: s.subs({x: 123}).simplify_full()) # doctest: +SKIP
    |a><a|
    [0 0]
    [0 0]
    >>> (V.H * V).apply_map(lambda s: s.subs({x: 123}).simplify_full()) # doctest: +SKIP
    |a><a|
    [1 0]
    [0 1]
