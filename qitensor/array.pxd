cimport cpython
cimport numpy as np

from qitensor.space cimport HilbertSpace

cdef class HilbertArray:
    cdef readonly HilbertSpace space
    cdef readonly list axes
    cdef public np.ndarray nparray

    cpdef copy(self)
    cpdef _reassign(self, HilbertArray other)
    cpdef get_dim(self, atom)
    cpdef _assert_same_axes(self, other)
    cpdef assert_density_matrix(self, check_hermitian=*, check_normalized=*, check_positive=*)
    cpdef set_data(self, new_data)
    cpdef tensordot(self, HilbertArray other, contraction_spaces=*)
    cpdef transpose(self, tpose_axes=*)
    cpdef relabel(self, from_spaces, to_spaces=*)
    cpdef relabel_prime(self)
    cpdef apply_map(self, fn)
    cpdef closeto(self, other, rtol=*, atol=*)
    cpdef lmul(self, other)
    cpdef _index_key_to_map(self, key)
    cpdef _get_set_item(self, key, do_set=*, set_val=*)
    cpdef _space_string(self, spc_set)
    cpdef _get_row_col_spaces(self, row_space=*, col_space=*)
    cpdef diag(self)
    cpdef as_np_matrix(self, dtype=*, row_space=*, col_space=*)
    cpdef np_matrix_transform(self, f, transpose_dims=*, row_space=*, col_space=*)
    cpdef det(self)
    cpdef fill(self, val)
    cpdef norm(self)
    cpdef trace_norm(self, row_space=*, col_space=*)
    cpdef schatten_norm(self, p, row_space=*, col_space=*)
    cpdef normalize(self)
    cpdef normalized(self)
    cpdef inv(self, row_space=*)
    cpdef conj(self)
    cpdef svd(self, full_matrices=*, inner_space=*)
    cpdef svd_list(self, row_space=*, col_space=*, thresh=*)
    cpdef singular_vals(self, row_space=*, col_space=*)
    cpdef eig(self, w_space=*, hermit=*)
    cpdef eigvals(self, hermit=*)
    cpdef sqrt(self)
    cpdef entropy(self, normalize=*, checks=*)
    cpdef purity(self, normalize=*, checks=*)
    cpdef QR(self, inner_space=*)
    cpdef tuple measure(self, HilbertSpace spc=*, cpython.bool normalize=*)
    cpdef span(self, axes=*)
    cpdef sage_matrix(self, R=*)
    cpdef sage_block_matrix(self, R=*)
    cpdef sage_matrix_transform(self, f, transpose_dims=*)
