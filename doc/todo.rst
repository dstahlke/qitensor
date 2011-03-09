TODO
====

finish docs

* docs for properties (array.space, atom.is_dual, array.nparray)
* fix sage doctests

features

* matrix pow
* trace
* eigenvalues
* quantum circuit operators (cnot, etc.)

fixes

* make sure pickle is done right
* cannot set slice data e.g. m[0:3] = [1,2,3]
* use singletons for HilbertAtom and HilbertBaseField (and make pickle restore the singletons)
* np.product returns float for empty list, and this is used extensively for computing dimensions
