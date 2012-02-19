Sage: Vectors Over the Symbolic Ring
====================================

    >>> from sage.all import * # doctest: +SKIP
    >>> from qitensor import qubit, set_qitensor_printoptions # doctest: +SKIP
    >>> # use Sage's nice array formatting
    >>> set_qitensor_printoptions(str_use_sage=True) # doctest: +SKIP

    >>> ha = qubit('a', dtype=SR) # doctest: +SKIP
    >>> hb = qubit('b', dtype=SR) # doctest: +SKIP
    >>> (x, y) = var('x y') # doctest: +SKIP
    >>> s = ha.array([1, x]) # doctest: +SKIP
    >>> t = hb.array([1, y]) # doctest: +SKIP
    >>> s*t # doctest: +SKIP
    HilbertArray(|a,b>,
    [  1]
    [  y]
    [---]
    [  x]
    [x*y])

    >>> # density operator
    >>> (s*t).O # doctest: +SKIP
    HilbertArray(|a,b><a,b|,
    [                            1                  conjugate(y)|                 conjugate(x)     conjugate(x)*conjugate(y)]
    [                            y                y*conjugate(y)|               y*conjugate(x)   y*conjugate(x)*conjugate(y)]
    [-----------------------------------------------------------+-----------------------------------------------------------]
    [                            x                x*conjugate(y)|               x*conjugate(x)   x*conjugate(x)*conjugate(y)]
    [                          x*y              x*y*conjugate(y)|             x*y*conjugate(x) x*y*conjugate(x)*conjugate(y)])
    >>> # density operator with ``ha`` traced out
    >>> (s*t).O.trace(ha) # doctest: +SKIP
    HilbertArray(|b><b|,
    [                            x*conjugate(x) + 1     x*conjugate(x)*conjugate(y) + conjugate(y)]
    [                          x*y*conjugate(x) + y x*y*conjugate(x)*conjugate(y) + y*conjugate(y)])
    >>> # there is no entanglement for this separable state
    >>> (s*t).O.trace(ha).entropy(normalize=True) # doctest: +SKIP
    0
    >>> # this however is an entangled state
    >>> (ha*hb).array([[1/sqrt(5),0],[0,2/sqrt(5)]]).O.trace(ha).entropy() # doctest: +SKIP
    -1/5*log(1/5)/log2 - 4/5*log(4/5)/log2
    >>> # The entropy from a symbolic expression is a big mess.  This expression
    >>> # has been verified numerically, but simplify_full() gives something
    >>> # which does not match numerically.
    >>> (ha*hb).array([[1,1],[1,exp(I*pi*x)]]).normalized().O.trace(ha).entropy(checks=False) # doctest: +SKIP
    1/4*((e^(I*pi*x) + 1)*sqrt(e^(I*pi*x)) - 2*e^(I*pi*x))*e^(-I*pi*x)*log(-1/4*((e^(I*pi*x) + 1)*sqrt(e^(I*pi*x)) - 2*e^(I*pi*x))*e^(-I*pi*x))/log2 - 1/4*((e^(I*pi*x) + 1)*sqrt(e^(I*pi*x)) + 2*e^(I*pi*x))*e^(-I*pi*x)*log(1/4*((e^(I*pi*x) + 1)*sqrt(e^(I*pi*x)) + 2*e^(I*pi*x))*e^(-I*pi*x))/log2

    >>> U = (ha * hb).eye() # doctest: +SKIP
    >>> # arrays can be indexed using dictionaries
    >>> U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 0 }] = x # doctest: +SKIP
    >>> U[{ ha: 0, ha.H: 0, hb: 0, hb.H: 1 }] = y # doctest: +SKIP
    >>> U # doctest: +SKIP
    HilbertArray(|a,b><a,b|,
    [x y|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1])

    >>> U.I # doctest: +SKIP
    HilbertArray(|a,b><a,b|,
    [ 1/x -y/x|   0    0]
    [   0    1|   0    0]
    [---------+---------]
    [   0    0|   1    0]
    [   0    0|   0    1])

    >>> U * U.I # doctest: +SKIP
    HilbertArray(|a,b><a,b|,
    [1 0|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1])

    >>> ((U ** 3) * (U ** -3)).simplify_full() # doctest: +SKIP
    HilbertArray(|a,b><a,b|,
    [1 0|0 0]
    [0 1|0 0]
    [---+---]
    [0 0|1 0]
    [0 0|0 1])

    >>> M = ha.O.array([[1,1],[1,x]]) # doctest: +SKIP
    >>> (W, V) = M.eig() # doctest: +SKIP
    >>> W # doctest: +SKIP
    HilbertArray(|a><a|,
    [1/2*x + 1/2*sqrt(x^2 - 2*x + 5) + 1/2                                     0]
    [                                    0 1/2*x - 1/2*sqrt(x^2 - 2*x + 5) + 1/2])
    >>> V # doctest: +SKIP
    HilbertArray(|a><a|,
    [                                1/sqrt(1/4*(x + sqrt(x^2 - 2*x + 5) - 1)^2 + 1)                                 1/sqrt(1/4*(x - sqrt(x^2 - 2*x + 5) - 1)^2 + 1)]
    [1/2*(x + sqrt(x^2 - 2*x + 5) - 1)/sqrt(1/4*(x + sqrt(x^2 - 2*x + 5) - 1)^2 + 1) 1/2*(x - sqrt(x^2 - 2*x + 5) - 1)/sqrt(1/4*(x - sqrt(x^2 - 2*x + 5) - 1)^2 + 1)])
    >>> (V.H * M * V - W).apply_map(lambda s: s.subs({x: 123})).simplify_full() # doctest: +SKIP
    HilbertArray(|a><a|,
    [0 0]
    [0 0])
    >>> (V.H * V).apply_map(lambda s: s.subs({x: 123})).simplify_full() # doctest: +SKIP
    HilbertArray(|a><a|,
    [1 0]
    [0 1])
