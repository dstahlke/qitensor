TODO
====

FINISH DOCS
    * better way to run Sage doctests?
    * ``sage -coverage qitensor/*.py``

BUGS
    * requires Cython 0.16.beta0 or there are segfaults (Sage only has 0.15.1)
      * for this reason, the spkg should contain the *.c files
    * redeclaring HilbertAtom can't change latex label
    * set formatting options on sage examples (including webpage)
    * detect when running under 'ipython qtconsole' and disable mathjax in html
    * sage doctests in python files give sphinx error
    * wrong exceptions are thrown
