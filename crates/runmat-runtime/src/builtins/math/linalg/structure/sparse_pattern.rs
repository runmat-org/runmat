//! Structural CSC traversal shared by sparse matrix metadata queries.
//!
//! The sparse tensor has already validated its column pointers and row indices.
//! Read its authoritative typed values: a floating mirror of wide integers is
//! not a valid source of zero/nonzero facts.

use runmat_value::{ComplexElement, SparseTensor};

pub(super) fn for_each_nonzero(sparse: &SparseTensor, mut visit: impl FnMut(usize, usize)) {
    for col in 0..sparse.cols {
        for index in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let nonzero = if sparse.is_logical() {
                true
            } else if let Some(values) = sparse.as_complex_f64_slice() {
                let ComplexElement(real, imaginary) = values[index];
                real != 0.0 || imaginary != 0.0
            } else if let Some(values) = sparse.as_complex_f32_slice() {
                let ComplexElement(real, imaginary) = values[index];
                real != 0.0 || imaginary != 0.0
            } else {
                sparse
                    .numeric_value_at(index)
                    .is_some_and(|value| !value.is_zero())
            };
            if nonzero {
                visit(sparse.row_indices[index], col);
            }
        }
    }
}
