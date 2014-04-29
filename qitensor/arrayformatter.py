"""
This module handles formatting of arrays.  Everything in here is for internal use only,
except for the :func:`set_qitensor_printoptions` and :func:`get_qitensor_printoptions`
functions.
"""

import numpy as np

from qitensor import have_sage
from qitensor.exceptions import HilbertError

__all__ = ['set_qitensor_printoptions', 'get_qitensor_printoptions', 'setup_qitensor_for_qtconsole', 'HilbertArrayFormatter']

class HilbertArrayFormatter(object):
    def __init__(self):
        """
        This module handles formatting of arrays.

        Methods of this class are called by methods of HilbertArray, and
        shouldn't need to be dealt with directly.

        sage: import qitensor.arrayformatter
        sage: TestSuite(qitensor.arrayformatter.FORMATTER).run()
        """

        self.str_use_sage = False
        # FIXME - make this undocumented option public (requires publishing np_colorizer)
        self.str_use_colorize = False
        self.zero_color_latex = 'Silver'
        self.zero_color_html = '#cccccc'
        self.use_latex_label_in_html = True
        self.ipy_table_format_mode = 'html'
        self.ipy_space_format_mode = 'latex'

    def _get_suppress(self):
        """
        Gets the current suppression settings (from numpy).
        """

        suppress = np.get_printoptions()['suppress']
        suppress_thresh = 0.1 ** (np.get_printoptions()['precision'] + 0.5)
        return (suppress, suppress_thresh)

    def py_scalar_latex_formatter(self, data, dollar_if_tex):
        """
        Formats python scalar for latex.
        """

        if data.dtype == complex:
            (suppress, suppress_thresh) = self._get_suppress()
            precision = np.get_printoptions()['precision']
            return np.core.arrayprint.ComplexFormat(
                data, precision=precision, suppress_small=suppress)
        else:
            return str

    def sage_scalar_latex_formatter(self, data, dollar_if_tex):
        """
        Formats Sage scalar for latex.
        """

        if not have_sage:
            raise HilbertError('This is only available under Sage')

        import sage.all

        if dollar_if_tex:
            return lambda x: '$'+sage.all.latex(x)+'$'
        else:
            return lambda x: sage.all.latex(x)

    def sympy_scalar_latex_formatter(self, data, dollar_if_tex):
        """
        Formats Sympy scalar for latex.
        """

        import sympy

        if dollar_if_tex:
            return lambda x: '$'+sympy.latex(x)+'$'
        else:
            return lambda x: sympy.latex(x)

    def _get_arr_obj(self, arr):
        if self.str_use_sage:
            return arr.sage_block_matrix()
        elif self.str_use_colorize:
            import np_colorizer
            return np_colorizer.colorize(arr.nparray)
        else:
            return arr.nparray

    def array_str(self, arr):
        """
        Creates string for HilbertArray.
        """

        return str(arr.space)+'\n'+str(self._get_arr_obj(arr))

    def array_repr(self, arr):
        """
        Creates repr for HilbertArray.
        """

        return 'HilbertArray('+repr(arr.space)+',\n'+repr(self._get_arr_obj(arr))+')'

    def array_latex_block_table(self, arr, use_hline=False):
        """
        Formats array in Latex.  Used by both Sage and IPython.
        """

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
                if suppress and spc.base_field.eval_suppress_small(v, suppress_thresh):
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
        r"""
        Format array in HTML.  Used for IPython.

        >>> from qitensor import qudit
        >>> ha = qudit('a', 3)
        >>> hb = qudit('b', 2)
        >>> X = ha.eye() * hb.ket(1)
        >>> f = HilbertArrayFormatter()
        >>> f.set_printoptions()
        >>> print(f.array_html_block_table(X))
        $\left| a,b \right\rangle\left\langle a \right|$<table style='margin: 0px 0px;'>
        <colgroup style='border: 2px solid black;'></colgroup>
        <colgroup span=3 style='border: 2px solid black;'></colgroup>
        <tbody style='border: 2px solid black;'>
        <tr style='border: 1px dotted; padding: 2px;'><td style='border: 1px dotted; padding: 2px; text-align: center;'> </td><td style='border: 1px dotted; padding: 2px; text-align: center;'><nobr>&#x27e8;<tt>0</tt>|</nobr></td><td style='border: 1px dotted; padding: 2px; text-align: center;'><nobr>&#x27e8;<tt>1</tt>|</nobr></td><td style='border: 1px dotted; padding: 2px; text-align: center;'><nobr>&#x27e8;<tt>2</tt>|</nobr></td></tr>
        </tbody>
        <tbody style='border: 2px solid black;'>
        <tr style='border: 1px dotted; padding: 2px;'><td style='border: 1px dotted; padding: 2px; text-align: center;'><nobr>|<tt>0</tt>,<tt>0</tt>&#x27e9;</nobr></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td></tr>
        <tr style='border: 1px dotted; padding: 2px;'><td style='border: 1px dotted; padding: 2px; text-align: center;'><nobr>|<tt>0</tt>,<tt>1</tt>&#x27e9;</nobr></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><nobr><tt> 1.+0.j</tt></nobr></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td></tr>
        </tbody>
        <tbody style='border: 2px solid black;'>
        <tr style='border: 1px dotted; padding: 2px;'><td style='border: 1px dotted; padding: 2px; text-align: center;'><nobr>|<tt>1</tt>,<tt>0</tt>&#x27e9;</nobr></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td></tr>
        <tr style='border: 1px dotted; padding: 2px;'><td style='border: 1px dotted; padding: 2px; text-align: center;'><nobr>|<tt>1</tt>,<tt>1</tt>&#x27e9;</nobr></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><nobr><tt> 1.+0.j</tt></nobr></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td></tr>
        </tbody>
        <tbody style='border: 2px solid black;'>
        <tr style='border: 1px dotted; padding: 2px;'><td style='border: 1px dotted; padding: 2px; text-align: center;'><nobr>|<tt>2</tt>,<tt>0</tt>&#x27e9;</nobr></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td></tr>
        <tr style='border: 1px dotted; padding: 2px;'><td style='border: 1px dotted; padding: 2px; text-align: center;'><nobr>|<tt>2</tt>,<tt>1</tt>&#x27e9;</nobr></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><font color='#cccccc'>0</font></td><td style='border: 1px dotted; padding: 2px; text-align: right;'><nobr><tt> 1.+0.j</tt></nobr></td></tr>
        </tbody>
        </table>
        <BLANKLINE>
        """

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

        ht = ''

        if self.use_latex_label_in_html:
            ht += '$'+spc._latex_()+'$'
        else:
            # FIXME - here, and elsewhere, use unicode symbols '&#x27e8;' and '&#x27e9;'
            # for html.
            ht += spc._html_()+'<br>'

        ht += "<table style='margin: 0px 0px;'>\n"

        if spc.ket_set:
            ht += "<colgroup "+st_tab+"></colgroup>\n"
        if len(spc.bra_set):
            colgrp_size = spc.bra_space().shape[-1]
            for i in range(spc.bra_space().dim() // colgrp_size):
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

                #if self.use_latex_label_in_html:
                #    ht += r'$\scriptsize{\left< '
                #    ht += ','.join([str(x) for x in b_idx]) # FIXME - latex label for indices?
                #    ht += r' \right|}$'
                #else:
                ht += '&#x27e8;'+(','.join(['<tt>'+str(x)+'</tt>' for x in b_idx]))+'|'

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

                #if self.use_latex_label_in_html:
                #    ht += r'$\scriptsize{\left| '
                #    ht += ','.join([str(x) for x in k_idx]) # FIXME - latex label for indices?
                #    ht += r' \right>}$'
                #else:
                ht += '|'+(','.join(['<tt>'+str(x)+'</tt>' for x in k_idx]))+'&#x27e9;'

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
                if suppress and spc.base_field.eval_suppress_small(v, suppress_thresh):
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

    # NOTE: this is normally accessed via set_qitensor_printoptions
    def set_printoptions(
        self,
        str_use_sage=None,
        zero_color_latex=None,
        zero_color_html=None,
        use_latex_label_in_html=None,
        ipy_table_format_mode=None,
        ipy_space_format_mode=None
    ):
        """
        Sets print options for qitensor.

        Any options passed the ``None`` value won't be changed.

        :param str_use_sage: If true, use Sage's matrix formatting functions
            when available (this is prettier).
        :type str_use_sage: bool

        :param zero_color_latex: Color to use for drawing the number zero in latex.
        :type zero_color_latex: string

        :param zero_color_html: Color to use for drawing the number zero in HTML.
        :type zero_color_html: string

        :param use_latex_label_in_html: If true, HilbertSpace labels will be
            shown in latex form when rendering an array in HTML.  Works good with
            the IPython notebook, but not with qtconsole.
        :type use_latex_label_in_html: bool

        :param ipy_table_format_mode: Which mode to use for formatting arrays in
            the IPython notebook.
        :type ipy_table_format_mode: string ('html', 'latex', 'png', 'plain')

        :param ipy_space_format_mode: Which mode to use for formatting HilbertSpace
            labels in the IPython notebook.
        :type ipy_space_format_mode: string ('latex', 'png', 'plain')

        qitensor also makes use of the ``suppress`` and ``precision`` options from
        numpy.set_printoptions.

        See also: :func:`get_qitensor_printoptions`
        """

        if str_use_sage is not None:
            self.str_use_sage = bool(str_use_sage)
        if zero_color_latex is not None:
            self.zero_color_latex = str(zero_color_latex)
        if zero_color_html is not None:
            self.zero_color_html = str(zero_color_html)
        if use_latex_label_in_html is not None:
            self.use_latex_label_in_html = bool(use_latex_label_in_html)
        if ipy_table_format_mode is not None:
            assert ipy_table_format_mode in ['html', 'latex', 'png', 'plain']
            self.ipy_table_format_mode = ipy_table_format_mode
        if ipy_space_format_mode is not None:
            assert ipy_space_format_mode in ['latex', 'png', 'plain']
            self.ipy_space_format_mode = ipy_space_format_mode

    # NOTE: this is normally accessed via get_qitensor_printoptions
    def get_printoptions(self):
        """
        Gets the current qitensor formatting options.

        See also: :func:`set_qitensor_printoptions`
        """

        return {
            "str_use_sage"            : self.str_use_sage,
            "zero_color_latex"        : self.zero_color_latex,
            "zero_color_html"         : self.zero_color_html,
            "use_latex_label_in_html" : self.use_latex_label_in_html,
            "ipy_table_format_mode"   : self.ipy_table_format_mode,
            "ipy_space_format_mode"   : self.ipy_space_format_mode,
        }

    def setup_for_qtconsole(self):
        """
        Sets good printing options for IPython QTconsole.
        """

        self.set_printoptions(ipy_table_format_mode='png', ipy_space_format_mode='png')
        # FIXME - latex_to_png is limited in its allowed colors
        self.set_printoptions(zero_color_latex='yellow')

FORMATTER = HilbertArrayFormatter()
set_qitensor_printoptions = FORMATTER.set_printoptions
get_qitensor_printoptions = FORMATTER.get_printoptions
setup_qitensor_for_qtconsole = FORMATTER.setup_for_qtconsole
