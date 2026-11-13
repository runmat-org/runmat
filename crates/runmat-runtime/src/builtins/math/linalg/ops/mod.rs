//! Linear algebra operations builtins.

pub(crate) mod callable_linear_operator;
pub(in crate::builtins::math::linalg) mod complex_division;
pub(crate) mod cross;
pub(crate) mod ctranspose;
pub(crate) mod dense_evd;
pub(crate) mod dense_factor;
pub(crate) mod dot;
pub(crate) mod iterative_preconditioner;
/// Borrowed typed dense/CSC operator shared by host iterative solvers.
pub mod linear_operator;
pub(crate) mod matrix_arithmetic;
pub(crate) mod pagemtimes;
pub(crate) mod pagetranspose;
pub(crate) mod sparse_factor;
/// Borrowed typed CSC operator boundary shared by sparse solvers and providers.
pub mod sparse_operator;
pub(crate) mod trace;
pub(crate) mod transpose;
pub(crate) mod triangular_operator;

pub use cross::cross_host_real_for_provider;
pub use dot::dot_host_complex_for_provider;
pub use dot::dot_host_real_for_provider;
pub use dot::kernel_error::DotKernelError;
pub use matrix_arithmetic::{mldivide_host_real_for_provider, mrdivide_host_real_for_provider};

type TransposedSparseValues<T> = (usize, usize, Vec<usize>, Vec<usize>, Vec<T>);

pub(super) fn is_vector_or_matrix_shape(shape: &[usize]) -> bool {
    shape.iter().skip(2).all(|&extent| extent == 1)
}

pub(super) fn transpose_sparse_tensor(
    sparse: runmat_value::SparseTensor,
    conjugate: bool,
) -> Result<runmat_value::SparseTensor, String> {
    if let Some(values) = sparse.as_complex_f32_slice() {
        let (rows, cols, col_ptrs, row_indices, values) =
            transpose_sparse_complex_values(&sparse, values, conjugate)?;
        return runmat_value::SparseTensor::from_host_complex_f32_buffers(
            rows,
            cols,
            col_ptrs.into(),
            row_indices.into(),
            runmat_value::HostComplexBuffer::from_elements(values),
        );
    }
    if let Some(values) = sparse.as_complex_f64_slice() {
        let (rows, cols, col_ptrs, row_indices, values) =
            transpose_sparse_complex_values(&sparse, values, conjugate)?;
        return runmat_value::SparseTensor::from_host_complex_buffers(
            rows,
            cols,
            col_ptrs.into(),
            row_indices.into(),
            runmat_value::HostComplexBuffer::from_elements(values),
        );
    }
    if sparse.is_logical() {
        let values = vec![(); sparse.nnz()];
        let (rows, cols, col_ptrs, row_indices, _) = transpose_sparse_values(&sparse, &values)?;
        return runmat_value::SparseTensor::new_logical(rows, cols, col_ptrs, row_indices);
    }
    if let Some(storage) = sparse.integer_storage() {
        return transpose_integer_sparse_tensor(&sparse, storage);
    }
    if let Some(values) = sparse.as_f32_slice() {
        let (rows, cols, col_ptrs, row_indices, values) = transpose_sparse_values(&sparse, values)?;
        return runmat_value::SparseTensor::new_f32(rows, cols, col_ptrs, row_indices, values);
    }
    let values = sparse
        .as_f64_slice()
        .ok_or_else(|| "SparseTensor has unsupported or inconsistent storage".to_string())?;
    let (rows, cols, col_ptrs, row_indices, values) = transpose_sparse_values(&sparse, values)?;
    runmat_value::SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
}

fn transpose_sparse_complex_values<T>(
    sparse: &runmat_value::SparseTensor,
    values: &[runmat_value::ComplexElement<T>],
    conjugate: bool,
) -> Result<TransposedSparseValues<runmat_value::ComplexElement<T>>, String>
where
    T: Copy + std::ops::Neg<Output = T>,
{
    let (rows, cols, col_ptrs, row_indices, mut values) = transpose_sparse_values(sparse, values)?;
    if conjugate {
        for value in &mut values {
            value.1 = -value.1;
        }
    }
    Ok((rows, cols, col_ptrs, row_indices, values))
}

fn transpose_sparse_values<T: Clone>(
    sparse: &runmat_value::SparseTensor,
    values: &[T],
) -> Result<TransposedSparseValues<T>, String> {
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
    sparse: &runmat_value::SparseTensor,
    storage: &runmat_value::IntegerStorage,
) -> Result<runmat_value::SparseTensor, String> {
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
    runmat_value::SparseTensor::new_integer(rows, cols, col_ptrs, row_indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{NumericDType, SparseTensor, Value};

    fn sparse_result(value: Value) -> SparseTensor {
        let Value::SparseTensor(sparse) = value else {
            panic!("expected sparse transpose result")
        };
        sparse
    }

    #[test]
    fn sparse_complex_transpose_family_preserves_class_and_conjugation() {
        for dtype in [NumericDType::F32, NumericDType::F64] {
            let source = match dtype {
                NumericDType::F32 => SparseTensor::new_complex_f32(
                    3,
                    2,
                    vec![0, 2, 3],
                    vec![0, 2, 1],
                    vec![(1.0, 2.0), (3.0, -4.0), (5.0, 6.0)],
                ),
                NumericDType::F64 => SparseTensor::new_complex(
                    3,
                    2,
                    vec![0, 2, 3],
                    vec![0, 2, 1],
                    vec![(1.0, 2.0), (3.0, -4.0), (5.0, 6.0)],
                ),
                _ => unreachable!(),
            }
            .expect("complex sparse input");
            let transpose = sparse_result(
                block_on(transpose::transpose_builtin(
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        Value::SparseTensor(source.clone()),
                    ]),
                ))
                .expect("transpose"),
            );
            let page = sparse_result(
                block_on(pagetranspose::pagetranspose_builtin(
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        Value::SparseTensor(source.clone()),
                    ]),
                ))
                .expect("pagetranspose"),
            );
            let conjugate = block_on(ctranspose::ctranspose_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::SparseTensor(source),
                ]),
            ))
            .expect("ctranspose");
            let conjugate = sparse_result(conjugate.value().clone());
            for result in [&transpose, &page, &conjugate] {
                assert_eq!(result.numeric_dtype(), Some(dtype));
                assert_eq!(result.shape(), vec![2, 3]);
                assert_eq!(result.col_ptrs, vec![0, 1, 2, 3]);
                assert_eq!(result.row_indices, vec![0, 1, 0]);
            }
            assert_eq!(transpose.complex_at(0, 0), Some((1.0, 2.0)));
            assert_eq!(transpose.complex_at(1, 1), Some((5.0, 6.0)));
            assert_eq!(transpose.complex_at(0, 2), Some((3.0, -4.0)));
            assert_eq!(page.complex_at(0, 2), Some((3.0, -4.0)));
            assert_eq!(conjugate.complex_at(0, 0), Some((1.0, -2.0)));
            assert_eq!(conjugate.complex_at(1, 1), Some((5.0, -6.0)));
            assert_eq!(conjugate.complex_at(0, 2), Some((3.0, 4.0)));
        }
    }

    #[test]
    fn transpose_sparse_tensor_rebuilds_csc_storage() {
        let sparse = SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![10.0, 30.0, 20.0])
            .expect("sparse");

        let transposed = transpose_sparse_tensor(sparse, false).expect("transpose");

        assert_eq!(transposed.rows, 2);
        assert_eq!(transposed.cols, 3);
        assert_eq!(transposed.col_ptrs, vec![0, 1, 2, 3]);
        assert_eq!(transposed.row_indices, vec![0, 1, 0]);
        assert_eq!(transposed.materialize_f64(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn transpose_sparse_tensor_preserves_native_single_storage() {
        let sparse =
            SparseTensor::new_f32(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![1.25, 3.5, 2.0])
                .expect("single sparse");
        let transposed = transpose_sparse_tensor(sparse, false).expect("transpose");
        assert_eq!(
            transposed.numeric_dtype(),
            Some(runmat_value::NumericDType::F32)
        );
        assert_eq!(transposed.shape(), vec![2, 3]);
        assert_eq!(transposed.as_f32_slice(), Some(&[1.25, 2.0, 3.5][..]));
    }

    #[test]
    fn transpose_sparse_tensor_preserves_logical_pattern_storage() {
        let sparse =
            SparseTensor::new_logical(3, 2, vec![0, 2, 3], vec![0, 2, 1]).expect("logical");
        let transposed = transpose_sparse_tensor(sparse, false).expect("transpose");
        assert!(transposed.is_logical());
        assert_eq!(transposed.shape(), vec![2, 3]);
        assert_eq!(transposed.col_ptrs, vec![0, 1, 2, 3]);
        assert_eq!(transposed.row_indices, vec![0, 1, 0]);
    }

    #[test]
    fn transpose_sparse_tensor_preserves_exact_uint64_values() {
        let sparse = SparseTensor::new_integer(
            3,
            2,
            vec![0, 2, 3],
            vec![0, 2, 1],
            runmat_value::IntegerStorage::U64(vec![u64::MAX, 7, 9]),
        )
        .expect("typed sparse");

        let transposed = transpose_sparse_tensor(sparse, false).expect("transpose");

        assert_eq!(transposed.rows, 2);
        assert_eq!(transposed.cols, 3);
        assert_eq!(
            transposed.integer_storage(),
            Some(&runmat_value::IntegerStorage::U64(vec![u64::MAX, 9, 7]))
        );
    }
}
