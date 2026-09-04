use runmat_value::{NumericStorage, SparseTensor, Tensor, Value};

use crate::BuiltinResult;

use super::operation::ExponentialOperation;

pub(super) fn evaluate(
    operation: ExponentialOperation,
    sparse: SparseTensor,
) -> BuiltinResult<Value> {
    if operation.preserves_sparse_zeros() {
        preserve_sparse(operation, sparse)
    } else {
        densify(operation, sparse)
    }
}

fn preserve_sparse(operation: ExponentialOperation, sparse: SparseTensor) -> BuiltinResult<Value> {
    let rows = sparse.rows;
    let cols = sparse.cols;
    let col_ptrs = sparse.col_ptrs.clone();
    let row_indices = sparse.row_indices.clone();
    let output = if let Some(values) = sparse.as_f64_slice() {
        SparseTensor::new(
            rows,
            cols,
            col_ptrs,
            row_indices,
            values
                .iter()
                .map(|&value| operation.apply_f64(value))
                .collect(),
        )
    } else if let Some(values) = sparse.as_f32_slice() {
        SparseTensor::new_f32(
            rows,
            cols,
            col_ptrs,
            row_indices,
            values
                .iter()
                .map(|&value| operation.apply_f32(value))
                .collect(),
        )
    } else if sparse.is_logical() {
        SparseTensor::new(
            rows,
            cols,
            col_ptrs,
            row_indices,
            vec![operation.apply_f64(1.0); sparse.nnz()],
        )
    } else if let Some(storage) = sparse.integer_storage() {
        super::host::ensure_exact_integer(operation, storage)?;
        SparseTensor::new(
            rows,
            cols,
            col_ptrs,
            row_indices,
            storage
                .to_f64_vec()
                .into_iter()
                .map(|value| operation.apply_f64(value))
                .collect(),
        )
    } else {
        return Err(super::errors::invalid(
            operation,
            "unsupported sparse storage",
        ));
    };
    output
        .map(Value::SparseTensor)
        .map_err(|error| super::errors::internal(operation, error))
}

fn densify(operation: ExponentialOperation, sparse: SparseTensor) -> BuiltinResult<Value> {
    let shape = vec![sparse.rows, sparse.cols];
    let output = if let Some(values) = sparse.as_f64_slice() {
        NumericStorage::F64(fill_dense(operation, &sparse, values, 0.0, |value| {
            operation.apply_f64(value)
        })?)
    } else if let Some(values) = sparse.as_f32_slice() {
        NumericStorage::F32(fill_dense(operation, &sparse, values, 0.0_f32, |value| {
            operation.apply_f32(value)
        })?)
    } else if sparse.is_logical() {
        let values = vec![1.0; sparse.nnz()];
        NumericStorage::F64(fill_dense(operation, &sparse, &values, 0.0, |value| {
            operation.apply_f64(value)
        })?)
    } else if let Some(storage) = sparse.integer_storage() {
        super::host::ensure_exact_integer(operation, storage)?;
        let values = storage.to_f64_vec();
        NumericStorage::F64(fill_dense(operation, &sparse, &values, 0.0, |value| {
            operation.apply_f64(value)
        })?)
    } else {
        return Err(super::errors::invalid(
            operation,
            "unsupported sparse storage",
        ));
    };
    Tensor::from_numeric_storage(output, shape)
        .map(Value::Tensor)
        .map_err(|error| super::errors::internal(operation, error))
}

fn fill_dense<T, F>(
    operation: ExponentialOperation,
    sparse: &SparseTensor,
    values: &[T],
    zero: T,
    apply: F,
) -> BuiltinResult<Vec<T>>
where
    T: Copy,
    F: Fn(T) -> T,
{
    if sparse.col_ptrs.len() != sparse.cols + 1 || sparse.row_indices.len() != values.len() {
        return Err(super::errors::internal(
            operation,
            "malformed sparse CSC storage",
        ));
    }
    let len = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
        super::errors::internal(operation, "sparse output element count overflow")
    })?;
    let mut dense = vec![apply(zero); len];
    for column in 0..sparse.cols {
        for index in sparse.col_ptrs[column]..sparse.col_ptrs[column + 1] {
            let row = *sparse
                .row_indices
                .get(index)
                .ok_or_else(|| super::errors::internal(operation, "malformed sparse row index"))?;
            let value = *values.get(index).ok_or_else(|| {
                super::errors::internal(operation, "malformed sparse value index")
            })?;
            let offset = row
                .checked_add(column.saturating_mul(sparse.rows))
                .filter(|offset| row < sparse.rows && *offset < dense.len())
                .ok_or_else(|| {
                    super::errors::internal(operation, "sparse row index out of bounds")
                })?;
            dense[offset] = apply(value);
        }
    }
    Ok(dense)
}
