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
* dtype=complex should still use Sage's rendering when in Sage

fixes

* make sure pickle is done right
* cannot set slice data e.g. m[0:3] = [1,2,3]
* use singletons for HilbertAtom and HilbertBaseField (and make pickle restore the singletons)
* np.product returns float for empty list, and this is used extensively for computing dimensions
* array creation in Sage needs to cast input data to type specified by sage_ring
* list of HilbertArray doesn't render proplerly in Sage jsMath
