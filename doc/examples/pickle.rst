Pickle
======

qitensor objects can be pickled.

    >>> from qitensor import qubit, indexed_space
    >>> import pickle
    >>> ha = qubit('a')
    >>> hb = indexed_space('b', ['x', 'y', 'z'])
    >>> x = (ha*hb).random_array()
    >>> s = pickle.dumps(x)
    >>> y = pickle.loads(s)
    >>> x == y
    True
    >>> x is y
    False
    >>> x.space is y.space
    True
