"""
FIXME - docstring
"""

import numpy as np

from qitensor import have_sage

__all__ = ['set_qitensor_printoptions']

class HilbertArrayFormatter(object):
    def __init__(self):
        self.str_use_sage = False
        self.repr_use_sage = False
        self.precision = 6
        self.suppress = True
        self.suppress_thresh = 1e-12
        self.zero_color_latex = 'Silver'

    def py_scalar_latex_formatter(self, data):
        if data.dtype == complex:
            # suppress=False here since supression is done elsewhere
            return np.core.arrayprint.ComplexFormat( \
                data, self.precision, False)
        else:
            return str

    def sage_scalar_latex_formatter(self, data):
        return lambda x: sage.all.latex(x)

    def array_str(self, arr):
        if self.str_use_sage:
            return str(arr.space)+'\n'+str(arr.sage_block_matrix())
        else:
            return str(arr.space)+'\n'+str(arr.nparray)

    def array_repr(self, arr):
        if self.repr_use_sage:
            return 'HilbertArray('+repr(arr.space)+',\n'+repr(arr.sage_block_matrix())+')'
        else:
            return 'HilbertArray('+repr(arr.space)+',\n'+repr(arr.nparray)+')'

    def array_latex_block_table(self, arr, use_hline=False):
        """Formats array in Latex.  Used by both Sage and IPython."""

# Alternative way to do it:
#        if not have_sage:
#            raise HilbertError('This is only available under Sage')
#
#        import sage.all
#
#        return '\\begin{array}{l}\n'+ \
#            sage.all.latex(self.space)+' \\\\\n'+ \
#            sage.all.latex(self.sage_block_matrix())+ \
#            '\\end{array}'

        spc = arr.space
        if len(spc.ket_set):
            ket_indices = list(spc.ket_space().index_iter())
        else:
            ket_indices = [None]
        if len(spc.bra_set):
            bra_indices = list(spc.bra_space().index_iter())
        else:
            bra_indices = [None]

        fmt = spc.base_field.latex_formatter(arr.nparray.flatten())

        ht = r'\scriptsize{'
        ht += r'\begin{array}{|'
        if spc.ket_set:
            ht += 'l|'

        if spc.bra_set:
            bra_shape = spc.bra_space().shape
            colgrp_size = np.product(bra_shape[1:])
            ht += ('c'*colgrp_size + '|')*bra_shape[0]
        else:
            ht += 'c|'
        ht += "}\n"

        if spc.bra_set:
            if use_hline: ht += r'\hline' + "\n"
            if spc.ket_set:
                ht += '&'
            for (b_idx_n, b_idx) in enumerate(bra_indices):
                if b_idx_n:
                    ht += ' & '
                if b_idx is not None:
                    ht += r'\mathbf{\left< '
                    for (x, y) in zip(b_idx, spc.sorted_bras):
                        ht += str(x) + '_{' + y.latex_label + '}'
                    ht += r' \right|}'
            ht += r' \\' + "\n"

        last_k = None
        for k_idx in ket_indices:
            if k_idx is None or k_idx[0] != last_k:
                if use_hline: ht += r'\hline' + "\n"
                if k_idx is not None:
                    last_k = k_idx[0]
            if k_idx is not None:
                ht += r'\mathbf{\left| '
                for (x, y) in zip(k_idx, spc.sorted_kets):
                    ht += str(x) + '_{' + y.latex_label + '}'
                ht += r' \right>}'
                ht += ' & '
            for (b_idx_n, b_idx) in enumerate(bra_indices):
                if k_idx is None and b_idx is None:
                    assert 0
                elif k_idx is None:
                    idx = b_idx
                elif b_idx is None:
                    idx = k_idx
                else:
                    idx = k_idx + b_idx
                v = arr[idx]
                if self.suppress and abs(v) < self.suppress_thresh:
                    if self.zero_color_latex:
                        vs = r'\color{'+self.zero_color_latex+'}{0}'
                    else:
                        vs = '0'
                else:
                    vs = fmt(v)
                if b_idx_n:
                    ht += ' & '
                ht += vs
            ht += r' \\' + "\n"

        if use_hline: ht += r'\hline' + "\n"
        ht += r"\end{array}" + "\n"
        ht += '}' # small

        return ht

    def array_html_block_table(self, arr):
        st_tab   = "style='border: 2px solid black;'"
        st_tr    = "style='border: 1px dotted; padding: 2px;'"
        st_th    = "style='border: 1px dotted; padding: 2px; text-align: center;'"
        st_tdval = "style='border: 1px dotted; padding: 2px; text-align: right;'"
        spc = arr.space
        if len(spc.ket_set):
            ket_indices = list(spc.ket_space().index_iter())
        else:
            ket_indices = [None]
        if len(spc.bra_set):
            bra_indices = list(spc.bra_space().index_iter())
        else:
            bra_indices = [None]
        # FIXME - if this really returns latex, then dollar signs need to be added
        fmt = spc.base_field.latex_formatter(arr.nparray.flatten())

        ht = "<table style='margin: 0px 0px;'>\n"

        if spc.ket_set:
            ht += "<colgroup "+st_tab+"></colgroup>\n"
        if len(spc.bra_set):
            colgrp_size = spc.bra_space().shape[-1]
            for i in range(spc.bra_space().dim() / colgrp_size):
                ht += ("<colgroup span=%d "+st_tab+"></colgroup>\n") % colgrp_size
        else:
            ht += "<colgroup "+st_tab+"></colgroup>\n"

        if spc.bra_set:
            ht += "<tbody "+st_tab+">\n"
            ht += '<tr '+st_tr+'>'
            if spc.ket_set:
                ht += '<td '+st_th+'> </td>'

            for b_idx in bra_indices:
                ht += '<td '+st_th+'><nobr>'
                #ht += r'$\left< '
                #for (x, y) in zip(b_idx, spc.sorted_bras):
                #    ht += str(x) + '_{' + y.latex_label + '}'
                #ht += r' \right|$'
                ht += '&lt;'
                ht += ','.join(y.label+'='+str(x) for (x, y) in zip(b_idx, spc.sorted_bras))
                ht += '|'
                ht += '</nobr></td>'

            ht += '</tr>\n'
            ht += '</tbody>\n'

        last_k = None
        for k_idx in ket_indices:
            if k_idx is not None and len(k_idx) > 1 and k_idx[-2] != last_k:
                if last_k is not None:
                    ht += '</tbody>\n'
                ht += "<tbody "+st_tab+">\n"
                last_k = k_idx[-2]
            ht += '<tr '+st_tr+'>'
            if spc.ket_set:
                ht += '<td '+st_th+'><nobr>'
                #ht += r'$\left| '
                #for (x, y) in zip(k_idx, spc.sorted_kets):
                #    ht += str(x) + '_{' + y.latex_label + '}'
                #ht += r' \right>$'
                ht += '|'
                ht += ','.join(y.label+'='+str(x) for (x, y) in zip(k_idx, spc.sorted_kets))
                ht += '&gt;'
                ht += '</nobr></td>'
            for b_idx in bra_indices:
                if k_idx is None and b_idx is None:
                    assert 0
                elif k_idx is None:
                    idx = b_idx
                elif b_idx is None:
                    idx = k_idx
                else:
                    idx = k_idx + b_idx
                v = arr[idx]
                if self.suppress and abs(v) < self.suppress_thresh:
                    vs = "<font color='#cccccc'>0</font>"
                else:
                    vs = "<nobr><tt>"+fmt(v)+"</tt></nobr>"
                ht += '<td '+st_tdval+'>'+vs+'</td>'
            ht += '</tr>\n'
        ht += '</tbody>\n'
        ht += '</table>\n'

        return ht

FORMATTER = HilbertArrayFormatter()

# FIXME - option for html vs. latex vs. none for ipython pretty printing
def set_qitensor_printoptions(precision=None, suppress=None, suppress_thresh=None):
    if precision is not None:
        FORMATTER.precision = precision
    if suppress is not None:
        FORMATTER.suppress = suppress
    if suppress_thresh is not None:
        FORMATTER.suppress_thresh = suppress_thresh
