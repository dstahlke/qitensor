cimport cpython
cimport numpy as np

cpdef _factory(dtype)

cdef class HilbertBaseField:
    cdef readonly dtype
    cdef readonly str unique_id
    cdef readonly sage_ring

    cpdef assert_same(self, HilbertBaseField other)
    cpdef matrix_np_to_sage(self, np.ndarray np_mat, R=*)
    cpdef np.ndarray matrix_sage_to_np(self, sage_mat)
    cpdef latex_formatter(self, data, dollar_if_tex)
    cpdef input_cast_function(self)
    cpdef complex_unit(self)
    cpdef infty(self)
    cpdef fractional_phase(self, int a, int b)
    cpdef frac(self, a, b)
    cpdef sqrt(self, x)
    cpdef log2(self, x)
    cpdef xlog2x(self, x)
    cpdef np.ndarray random_array(self, shape)
    cpdef np.ndarray eye(self, long size)
    cpdef np.ndarray mat_adjoint(self, np.ndarray mat)
    cpdef np.ndarray mat_inverse(self, np.ndarray mat)
    cpdef mat_det(self, np.ndarray mat)
    cpdef mat_norm(self, np.ndarray arr, p)
    cpdef np.ndarray mat_pinv(self, np.ndarray mat, rcond)
    cpdef np.ndarray mat_conj(self, np.ndarray mat)
    cpdef np.ndarray mat_n(self, np.ndarray mat, prec=*, digits=*)
    cpdef np.ndarray mat_simplify(self, np.ndarray mat, full=*)
    cpdef np.ndarray mat_expm(self, np.ndarray mat)
    cpdef np.ndarray mat_logm(self, np.ndarray mat)
    cpdef np.ndarray mat_pow(self, np.ndarray mat, n)
    cpdef mat_svd(self, np.ndarray mat, full_matrices)
    cpdef mat_svd_vals(self, np.ndarray mat)
    cpdef mat_eig(self, np.ndarray mat, cpython.bool hermit)
    cpdef mat_eigvals(self, np.ndarray mat, cpython.bool hermit)
    cpdef mat_qr(self, np.ndarray mat)
    cpdef eval_suppress_small(self, x, float threshold)
