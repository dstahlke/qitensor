#!/bin/sh

fatal() {
	echo Fatal error.
	exit 1
}

test "z" == "z$1" && fatal

rst_fn=$1
echo "Testing $rst_fn"

sage_fn=sagetest-$rst_fn
test -e $rst_fn || fatal

rm -f $sage_fn
(echo :: ; grep '^ ' $rst_fn |perl -p -e 's/ # docte.*//; s/>>>/sage:/') > $sage_fn
test -e $sage_fn || fatal

sage -t $sage_fn
rm -f $sage_fn
