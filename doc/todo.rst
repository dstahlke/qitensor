TODO
====

finish docs

* docs for properties (array.space, atom.is_dual, array.nparray)
* better way to run Sage doctests?

features

* matrix pow
* trace
* eigenvalues

fixes

* make sure pickle is done right
* use singletons for HilbertAtom and HilbertBaseField (and make pickle restore the singletons)
* np.product returns float for empty list, and this is used extensively for computing dimensions
