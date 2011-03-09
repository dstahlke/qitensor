Pickle
======

qitensor objects can be pickled.  However, the internals are currently in flux,
so it is possible that stored objects might not be compatible with future
versions.

    >>> from qitensor import qubit
    >>> import pickle
    >>> ha = qubit('a')
    >>> x = ha.array([1, 2])
    >>> s = pickle.dumps(x)
    >>> pickle.loads(s)
    HilbertArray(|a>,
    array([ 1.+0.j,  2.+0.j]))
