cpdef _factory(dtype)

cdef class HilbertBaseField:
    cdef readonly dtype
    cdef readonly unique_id
    cdef readonly sage_ring

    cpdef assert_same(self, HilbertBaseField other)
    cpdef matrix_np_to_sage(self, np_mat, R=*)
    cpdef matrix_sage_to_np(self, sage_mat)
    cpdef latex_formatter(self, data, dollar_if_tex)
    cpdef input_cast_function(self)
    cpdef complex_unit(self)
    cpdef fractional_phase(self, a, b)
    cpdef sqrt(self, x)
    cpdef xlog2x(self, x)
    cpdef random_array(self, shape)
    cpdef eye(self, size)
    cpdef mat_adjoint(self, mat)
    cpdef mat_inverse(self, mat)
    cpdef mat_det(self, mat)
    cpdef mat_norm(self, arr)
    cpdef mat_pinv(self, mat, rcond)
    cpdef mat_conj(self, mat)
    cpdef mat_n(self, mat, prec=*, digits=*)
    cpdef mat_simplify(self, mat, full=*)
    cpdef mat_expm(self, mat, q)
    cpdef mat_pow(self, mat, n)
    cpdef mat_svd(self, mat, full_matrices)
    cpdef mat_svd_vals(self, mat)
    cpdef mat_eig(self, mat, hermit)
    cpdef mat_eigvals(self, mat, hermit)
    cpdef mat_qr(self, mat)
