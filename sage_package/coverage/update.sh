cp ../../qitensor/*.{py,pyx} .
perl -pi -e 's/>>>/sage:/' *.{py,pyx}
sage -coverage *.{py,pyx}
