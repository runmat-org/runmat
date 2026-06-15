//! Linear algebra operations builtins.

pub(crate) mod cross;
pub(crate) mod ctranspose;
pub(crate) mod dot;
pub(crate) mod mldivide;
pub(crate) mod mpower;
pub(crate) mod mrdivide;
pub(crate) mod mtimes;
pub(crate) mod trace;
pub(crate) mod transpose;

pub use cross::cross_host_real_for_provider;
pub use dot::dot_host_complex_for_provider;
pub use dot::dot_host_real_for_provider;
pub use mldivide::mldivide_host_real_for_provider;
pub use mrdivide::mrdivide_host_real_for_provider;

pub(super) fn transpose_real_sparse_tensor(
    sparse: runmat_builtins::SparseTensor,
) -> Result<runmat_builtins::SparseTensor, String> {
    let mut triplets = Vec::with_capacity(sparse.nnz());
    for col in 0..sparse.cols {
        for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            triplets.push((col, sparse.row_indices[idx], sparse.values[idx]));
        }
    }
    triplets.sort_by_key(|&(row, col, _)| (col, row));

    let rows = sparse.cols;
    let cols = sparse.rows;
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::with_capacity(triplets.len());
    let mut values = Vec::with_capacity(triplets.len());
    col_ptrs.push(0);
    let mut next = 0usize;
    for col in 0..cols {
        while next < triplets.len() && triplets[next].1 == col {
            row_indices.push(triplets[next].0);
            values.push(triplets[next].2);
            next += 1;
        }
        col_ptrs.push(values.len());
    }
    runmat_builtins::SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::SparseTensor;

    #[test]
    fn transpose_real_sparse_tensor_rebuilds_csc_storage() {
        let sparse = SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![10.0, 30.0, 20.0])
            .expect("sparse");

        let transposed = transpose_real_sparse_tensor(sparse).expect("transpose");

        assert_eq!(transposed.rows, 2);
        assert_eq!(transposed.cols, 3);
        assert_eq!(transposed.col_ptrs, vec![0, 1, 2, 3]);
        assert_eq!(transposed.row_indices, vec![0, 1, 0]);
        assert_eq!(transposed.values, vec![10.0, 20.0, 30.0]);
    }
}
