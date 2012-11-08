TODO
====

FINISH DOCS
    * better way to run Sage doctests?
    * ``sage -coverage qitensor/*.py``

BUGS/FEATURES
    * 'assert' doesn't give useful message now that Cython is being used
    * redeclaring HilbertAtom can't change latex label
    * set formatting options on sage examples (including webpage)
    * detect when running under 'ipython qtconsole' call setup_qitensor_for_qtconsole()
    * sage doctests in python files give sphinx error
    * wrong exceptions are thrown
    * printing large array is slow (time taken in array_html_block_table)
    * row_space option for expm, logm, det, pinv, etc.
    * use HilbertArray.closeto() instead of np.allclose()
