#!/bin/sh

fatal() {
	echo Fatal error.
	exit 1
}

ver=$(PYTHONPATH="..:$PYTHONPATH" python -c 'import qitensor; print qitensor.__version__') || fatal
build="qitensor-$ver"
mkdir -p $build || fatal
cp SPKG.txt spkg-install $build || fatal
rsync -av --delete .. $build/src \
	--exclude $build \
	--exclude .git \
	--exclude dist \
	--exclude build \
	--exclude doc/_templates \
	--exclude doc/_static \
	--exclude doc/_build \
	--exclude "*.pyc" \
	--exclude "*.spkg" \
	--exclude "*/.*.swp" \
	|| fatal
sage --spkg $build || fatal
ls -la $build.spkg || fatal
echo Done.
