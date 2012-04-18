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

Test of long-term storage:

    >>> x = pickle.loads('\x80\x02cqitensor.array\n_unreduce_v1\nq\x00cqitensor.space\n_unreduce_v1\nq\x01c__builtin__\nfrozenset\nq\x02]q\x03(cqitensor.atom\n_unreduce_v1\nq\x04(U\x01aq\x05h\x05K\x00K\x01\x86q\x06cqitensor.factory\nGroupOpCyclic_factory\nq\x07K\x02\x85q\x08Rq\tcqitensor.basefield\n_unreduce_v1\nq\nc__builtin__\ncomplex\nq\x0b\x85q\x0cRq\r\x89Ntq\x0eRq\x0fh\x04(U\x01bq\x10h\x10U\x01xq\x11U\x01yq\x12U\x01zq\x13\x87q\x14cqitensor.factory\nGroupOpTimes_factory\nq\x15)Rq\x16h\r\x89Ntq\x17Rq\x18e\x85q\x19Rq\x1ah\x02]q\x1b\x85q\x1cRq\x1d\x86q\x1eRq\x1fcnumpy.core.multiarray\n_reconstruct\nq cnumpy\nndarray\nq!K\x00\x85q"h\x10\x87q#Rq$(K\x01K\x02K\x03\x86q%cnumpy\ndtype\nq&U\x03c16q\'K\x00K\x01\x87q(Rq)(K\x03U\x01<q*NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq+b\x89U`\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x00\x00q,tq-b\x86q.Rq/.')
    >>> y = (ha*hb).array([[1,2,3],[4,5,6]])
    >>> x == y
    True
    >>> x is y
    False
    >>> x.space is y.space
    True
