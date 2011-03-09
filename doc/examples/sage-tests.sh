#!/bin/sh

(echo :: ; grep '^ ' group_indices.rst |perl -p -e 's/ # docte.*//; s/>>>/sage:/') > tmp1.rst
sage -t tmp1.rst
(echo :: ; grep '^ ' symbolic.rst |perl -p -e 's/ # docte.*//; s/>>>/sage:/') > tmp2.rst
sage -t tmp2.rst
rm tmp[12].rst
