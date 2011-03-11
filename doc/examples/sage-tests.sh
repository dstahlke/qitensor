#!/bin/sh

fatal() {
	echo Fatal error.
	exit 1
}

run_test() {
	rst_fn=$1
	echo "Testing $rst_fn"

	sage_fn=sage-$rst_fn
	test -e $rst_fn || fatal

	(echo :: ; grep '^ ' $rst_fn |perl -p -e 's/ # docte.*//; s/>>>/sage:/') > $sage_fn
	test -e $sage_fn || fatal

	sage -t $sage_fn
	rm -f $sage_fn
}

run_test group_indices.rst
run_test symbolic.rst
