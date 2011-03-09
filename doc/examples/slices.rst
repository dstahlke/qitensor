Slices and Views
================

    >>> from qitensor import qubit
    >>> ha = qubit('a')
    >>> hb = qubit('b')
    >>> ha * hb
    |a,b>
    >>> x = (ha * hb).array()
    >>> x
    HilbertArray(|a,b>,
    array([[ 0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j]]))

You can use numpy style indexes:

    >>> x[0, 0]
    0j

Or you can index using a dictionary:

    >>> x[{ ha: 0, hb: 0 }]
    0j

An incomplete index set returns a slice:

    >>> y = x[{ ha: 0 }]
    >>> y
    HilbertArray(|b>,
    array([ 0.+0.j,  0.+0.j]))

As in numpy, slices are views and so tie to the source array:
(note: y = [1, 2] would untie y)

    >>> y[:] = [1, 2]
    >>> y
    HilbertArray(|b>,
    array([ 1.+0.j,  2.+0.j]))
    >>> x
    HilbertArray(|a,b>,
    array([[ 1.+0.j,  2.+0.j],
           [ 0.+0.j,  0.+0.j]]))

This provides a convenient way to fill arrays:

    >>> x[{ ha: 1 }] = [3, 4]
    >>> x
    HilbertArray(|a,b>,
    array([[ 1.+0.j,  2.+0.j],
           [ 3.+0.j,  4.+0.j]]))

numpy style slices work the same, except it may not always be clear which axes
go to which component spaces:

    >>> x[0, :]
    HilbertArray(|b>,
    array([ 1.+0.j,  2.+0.j]))
    >>> x[:, 0]
    HilbertArray(|a>,
    array([ 1.+0.j,  3.+0.j]))

The .H property turns a ket space into a bra space (and vice versa):

    >>> ha.H
    <a|
    >>> ha.H.H
    |a>
    >>> (ha*hb).H
    <a,b|
    >>> ha * hb.H
    |a><b|

The .O property is a shortcut for (self * self.H).

    >>> (ha * hb).O
    |a,b><a,b|

To index a bra space:

    >>> x = ha.O.array()
    >>> x
    HilbertArray(|a><a|,
    array([[ 0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j]]))
    >>> x[{ ha: 0, ha.H: 1 }] = 5
    >>> x
    HilbertArray(|a><a|,
    array([[ 0.+0.j,  5.+0.j],
           [ 0.+0.j,  0.+0.j]]))

The .space attribute gives the space for an array.

    >>> x.space
    |a><a|

Spaces of larger dimension are supported.

    >>> from qitensor import qudit
    >>> hc = qudit('c', 3)
    >>> hc.array()
    HilbertArray(|c>,
    array([ 0.+0.j,  0.+0.j,  0.+0.j]))

It is also possible to index using something other than natural numbers.

    >>> from qitensor import indexed_space
    >>> hd = indexed_space('d', ['up', 'down'])
    >>> x = hd.array([5, 7])
    >>> x['up']
    (5+0j)
    >>> y = (ha * hd.H).array([[1, 2], [3, 4]])
    >>> y
    HilbertArray(|a><d|,
    array([[ 1.+0.j,  2.+0.j],
           [ 3.+0.j,  4.+0.j]]))
    >>> y[{ ha: 1, hd.H: 'down' }]
    (4+0j)
