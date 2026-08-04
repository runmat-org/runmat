//! Linear algebra operations builtins.

pub(crate) mod cross;
pub(crate) mod ctranspose;
pub(crate) mod dot;
pub(crate) mod mldivide;
pub(crate) mod mpower;
pub(crate) mod mrdivide;
pub(crate) mod mtimes;
pub(crate) mod pagemtimes;
pub(crate) mod pagetranspose;
pub(crate) mod trace;
pub(crate) mod transpose;

pub use cross::cross_host_real_for_provider;
pub use dot::dot_host_complex_for_provider;
pub use dot::dot_host_real_for_provider;
pub use mldivide::mldivide_host_real_for_provider;
pub use mrdivide::mrdivide_host_real_for_provider;

pub(super) fn is_vector_or_matrix_shape(shape: &[usize]) -> bool {
    shape.iter().skip(2).all(|&extent| extent == 1)
}

pub(super) fn transpose_real_sparse_tensor(
    sparse: runmat_builtins::SparseTensor,
) -> Result<runmat_builtins::SparseTensor, String> {
    if let Some(storage) = sparse.integer_storage() {
        return transpose_integer_sparse_tensor(&sparse, storage);
    }
    if let Some(values) = sparse.as_f32_slice() {
        let (rows, cols, col_ptrs, row_indices, values) = transpose_sparse_values(&sparse, values)?;
        return runmat_builtins::SparseTensor::new_f32(rows, cols, col_ptrs, row_indices, values);
    }
    let values = sparse.as_f64_slice().expect("double sparse storage");
    let (rows, cols, col_ptrs, row_indices, values) = transpose_sparse_values(&sparse, values)?;
    runmat_builtins::SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
}

fn transpose_sparse_values<T: Clone>(
    sparse: &runmat_builtins::SparseTensor,
    values: &[T],
) -> Result<(usize, usize, Vec<usize>, Vec<usize>, Vec<T>), String> {
    if values.len() != sparse.nnz() {
        return Err("SparseTensor value storage is inconsistent".to_string());
    }
    let mut triplets = Vec::with_capacity(sparse.nnz());
    for col in 0..sparse.cols {
        for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            triplets.push((col, sparse.row_indices[idx], values[idx].clone()));
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
            values.push(triplets[next].2.clone());
            next += 1;
        }
        col_ptrs.push(values.len());
    }
    Ok((rows, cols, col_ptrs, row_indices, values))
}

fn transpose_integer_sparse_tensor(
    sparse: &runmat_builtins::SparseTensor,
    storage: &runmat_builtins::IntegerStorage,
) -> Result<runmat_builtins::SparseTensor, String> {
    let mut triplets = Vec::with_capacity(sparse.nnz());
    for col in 0..sparse.cols {
        for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let value = storage
                .value_at(idx)
                .ok_or_else(|| "SparseTensor integer storage is inconsistent".to_string())?;
            triplets.push((col, sparse.row_indices[idx], value));
        }
    }
    triplets.sort_by_key(|(row, col, _)| (*col, *row));

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
            values.push(triplets[next].2.clone());
            next += 1;
        }
        col_ptrs.push(values.len());
    }
    let values = storage.from_same_class_values(values)?;
    runmat_builtins::SparseTensor::new_integer(rows, cols, col_ptrs, row_indices, values)
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
        assert_eq!(transposed.materialize_f64(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn transpose_real_sparse_tensor_preserves_native_single_storage() {
        let sparse =
            SparseTensor::new_f32(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![1.25, 3.5, 2.0])
                .expect("single sparse");
        let transposed = transpose_real_sparse_tensor(sparse).expect("transpose");
        assert_eq!(
            transposed.numeric_dtype(),
            runmat_builtins::NumericDType::F32
        );
        assert_eq!(transposed.shape(), vec![2, 3]);
        assert_eq!(transposed.as_f32_slice(), Some(&[1.25, 2.0, 3.5][..]));
    }

    #[test]
    fn transpose_real_sparse_tensor_preserves_exact_uint64_values() {
        let sparse = SparseTensor::new_integer(
            3,
            2,
            vec![0, 2, 3],
            vec![0, 2, 1],
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX, 7, 9]),
        )
        .expect("typed sparse");

        let transposed = transpose_real_sparse_tensor(sparse).expect("transpose");

        assert_eq!(transposed.rows, 2);
        assert_eq!(transposed.cols, 3);
        assert_eq!(
            transposed.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::U64(vec![u64::MAX, 9, 7]))
        );
    }
}
