"""
FIXME - docstring
"""

import numpy as np

from qitensor import have_sage

__all__ = ['set_qitensor_printoptions', 'get_qitensor_printoptions']

class HilbertArrayFormatter(object):
    def __init__(self):
        self.str_use_sage = False
        self.repr_use_sage = False
        self.zero_color_latex = 'Silver'
        self.zero_color_html = '#cccccc'
        self.use_latex_label_in_html = True
        self.ipy_table_format_mode = 'html'
        self.ipy_space_format_mode = 'latex'

    def _get_suppress(self):
        suppress = np.get_printoptions()['suppress']
        suppress_thresh = 0.1 ** (np.get_printoptions()['precision'] + 0.5)
        return (suppress, suppress_thresh)

    def py_scalar_latex_formatter(self, data, dollar_if_tex):
        if data.dtype == complex:
            precision = np.get_printoptions()['precision']
            # suppress=False here since supression is done elsewhere
            return np.core.arrayprint.ComplexFormat( \
                data, precision=precision, suppress_small=False)
        else:
            return str

    def sage_scalar_latex_formatter(self, data, dollar_if_tex):
        if not have_sage:
            raise HilbertError('This is only available under Sage')

        import sage.all

        if dollar_if_tex:
            return lambda x: '$'+sage.all.latex(x)+'$'
        else:
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

        (suppress, suppress_thresh) = self._get_suppress()

        spc = arr.space
        if len(spc.ket_set):
            ket_indices = list(spc.ket_space().index_iter())
        else:
            ket_indices = [None]
        if len(spc.bra_set):
            bra_indices = list(spc.bra_space().index_iter())
        else:
            bra_indices = [None]

        fmt = spc.base_field.latex_formatter(arr.nparray.flatten(), dollar_if_tex=False)

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
                    ht += r'\left< '
                    for (x, y) in zip(b_idx, spc.sorted_bras):
                        ht += str(x) + '_{' + y.latex_label + '}'
                    ht += r' \right|'
            ht += r' \\' + "\n"

        last_k = None
        for k_idx in ket_indices:
            if k_idx is None or k_idx[0] != last_k:
                if use_hline: ht += r'\hline' + "\n"
                if k_idx is not None:
                    last_k = k_idx[0]
            if k_idx is not None:
                ht += r'\left| '
                for (x, y) in zip(k_idx, spc.sorted_kets):
                    ht += str(x) + '_{' + y.latex_label + '}'
                ht += r' \right>'
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
                if suppress and abs(v) < suppress_thresh:
                    if self.zero_color_latex != '':
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
        (suppress, suppress_thresh) = self._get_suppress()

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
        fmt = spc.base_field.latex_formatter(arr.nparray.flatten(), dollar_if_tex=True)

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
                if self.use_latex_label_in_html:
                    ht += r'$\scriptsize{\left< '
                    for (x, y) in zip(b_idx, spc.sorted_bras):
                        ht += str(x) + '_{' + y.latex_label + '}'
                    ht += r' \right|}$'
                else:
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
                if self.use_latex_label_in_html:
                    ht += r'$\scriptsize{\left| '
                    for (x, y) in zip(k_idx, spc.sorted_kets):
                        ht += str(x) + '_{' + y.latex_label + '}'
                    ht += r' \right>}$'
                else:
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
                if suppress and abs(v) < suppress_thresh:
                    if self.zero_color_html != '':
                        vs = "<font color='"+self.zero_color_html+"'>0</font>"
                    else:
                        vs = "0"
                else:
                    vs = "<nobr><tt>"+fmt(v)+"</tt></nobr>"
                ht += '<td '+st_tdval+'>'+vs+'</td>'
            ht += '</tr>\n'
        ht += '</tbody>\n'
        ht += '</table>\n'

        return ht

    def set_printoptions(
        self,
        repr_use_sage=None,
        str_use_sage=None,
        zero_color_latex=None,
        zero_color_html=None,
        use_latex_label_in_html=None,
        ipy_table_format_mode=None,
        ipy_space_format_mode=None
    ):
        if repr_use_sage is not None:
            self.repr_use_sage  = bool(repr_use_sage)
        if str_use_sage is not None:
            self.str_use_sage = bool(str_use_sage)
        if zero_color_latex is not None:
            self.zero_color_latex = str(zero_color_latex)
        if zero_color_html is not None:
            self.zero_color_html = str(zero_color_html)
        if use_latex_label_in_html is not None:
            self.use_latex_label_in_html = bool(use_latex_label_in_html)
        if ipy_table_format_mode is not None:
            assert ipy_table_format_mode in ['html', 'latex', 'plain']
            self.ipy_table_format_mode = ipy_table_format_mode
        if ipy_space_format_mode is not None:
            assert ipy_space_format_mode in ['latex', 'plain']
            self.ipy_space_format_mode = ipy_space_format_mode

    def get_printoptions(self):
        return {
            "str_use_sage"            : self.str_use_sage,
            "repr_use_sage"           : self.repr_use_sage,
            "zero_color_latex"        : self.zero_color_latex,
            "zero_color_html"         : self.zero_color_html,
            "use_latex_label_in_html" : self.use_latex_label_in_html,
            "ipy_table_format_mode"   : self.ipy_table_format_mode,
            "ipy_space_format_mode"   : self.ipy_space_format_mode,
        }

FORMATTER = HilbertArrayFormatter()
set_qitensor_printoptions = FORMATTER.set_printoptions
get_qitensor_printoptions = FORMATTER.get_printoptions
