use crate::indexing::integer_assignment::{self, IntegerAssignmentValue};
use crate::indexing::write_slice::{
    delete_integer_complex_storage_positions, deleted_vector_shape,
};
use crate::interpreter::errors::mex;
use runmat_builtins::{ComplexTensor, IntValue, IntegerStorage, SparseTensor, Tensor, Value};
use runmat_runtime::RuntimeError;

fn map_assignment_shape_error(err: impl std::fmt::Display) -> RuntimeError {
    mex("ShapeMismatch", &format!("assignment: {err}"))
}

fn map_acceleration_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    mex("AccelerationOperationFailed", &format!("{context}: {err}"))
}

fn is_empty_tensor(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if t.data.is_empty() || t.rows == 0 || t.cols == 0)
        || matches!(value, Value::ComplexTensor(t) if t.data.is_empty() || t.rows == 0 || t.cols == 0)
}

fn integer_storage_scalar(storage: &IntegerStorage) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[0]),
        IntegerStorage::I16(values) => IntValue::I16(values[0]),
        IntegerStorage::I32(values) => IntValue::I32(values[0]),
        IntegerStorage::I64(values) => IntValue::I64(values[0]),
        IntegerStorage::U8(values) => IntValue::U8(values[0]),
        IntegerStorage::U16(values) => IntValue::U16(values[0]),
        IntegerStorage::U32(values) => IntValue::U32(values[0]),
        IntegerStorage::U64(values) => IntValue::U64(values[0]),
    }
}

async fn rhs_to_integer_assignment_scalar(
    rhs: &Value,
) -> Result<IntegerAssignmentValue, RuntimeError> {
    match rhs {
        Value::Int(value) => Ok(IntegerAssignmentValue::Exact(value.clone())),
        Value::Num(value) => Ok(IntegerAssignmentValue::Float(*value)),
        Value::Bool(value) => Ok(IntegerAssignmentValue::Float(if *value {
            1.0
        } else {
            0.0
        })),
        Value::Tensor(tensor) if tensor.data.len() == 1 => match tensor.integer_storage() {
            Some(storage) => Ok(IntegerAssignmentValue::Exact(integer_storage_scalar(
                storage,
            ))),
            None => Ok(IntegerAssignmentValue::Float(tensor.data[0])),
        },
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(IntegerAssignmentValue::Float(if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }))
        }
        _ => rhs_to_real_scalar(rhs)
            .await
            .map(IntegerAssignmentValue::Float),
    }
}

fn assign_integer_storage(
    storage: IntegerStorage,
    index: usize,
    target_len: usize,
    rhs: &IntegerAssignmentValue,
) -> IntegerStorage {
    let value = integer_assignment::scalar_value(&storage, rhs);
    match storage {
        IntegerStorage::I8(mut values) => {
            let IntValue::I8(value) = value else {
                unreachable!()
            };
            values.resize(target_len, 0);
            values[index] = value;
            IntegerStorage::I8(values)
        }
        IntegerStorage::I16(mut values) => {
            let IntValue::I16(value) = value else {
                unreachable!()
            };
            values.resize(target_len, 0);
            values[index] = value;
            IntegerStorage::I16(values)
        }
        IntegerStorage::I32(mut values) => {
            let IntValue::I32(value) = value else {
                unreachable!()
            };
            values.resize(target_len, 0);
            values[index] = value;
            IntegerStorage::I32(values)
        }
        IntegerStorage::I64(mut values) => {
            let IntValue::I64(value) = value else {
                unreachable!()
            };
            values.resize(target_len, 0);
            values[index] = value;
            IntegerStorage::I64(values)
        }
        IntegerStorage::U8(mut values) => {
            let IntValue::U8(value) = value else {
                unreachable!()
            };
            values.resize(target_len, 0);
            values[index] = value;
            IntegerStorage::U8(values)
        }
        IntegerStorage::U16(mut values) => {
            let IntValue::U16(value) = value else {
                unreachable!()
            };
            values.resize(target_len, 0);
            values[index] = value;
            IntegerStorage::U16(values)
        }
        IntegerStorage::U32(mut values) => {
            let IntValue::U32(value) = value else {
                unreachable!()
            };
            values.resize(target_len, 0);
            values[index] = value;
            IntegerStorage::U32(values)
        }
        IntegerStorage::U64(mut values) => {
            let IntValue::U64(value) = value else {
                unreachable!()
            };
            values.resize(target_len, 0);
            values[index] = value;
            IntegerStorage::U64(values)
        }
    }
}

fn delete_from_integer_storage(storage: IntegerStorage, index: usize) -> IntegerStorage {
    macro_rules! delete_storage {
        ($values:expr, $variant:ident) => {{
            let mut values = $values;
            values.remove(index);
            IntegerStorage::$variant(values)
        }};
    }

    match storage {
        IntegerStorage::I8(values) => delete_storage!(values, I8),
        IntegerStorage::I16(values) => delete_storage!(values, I16),
        IntegerStorage::I32(values) => delete_storage!(values, I32),
        IntegerStorage::I64(values) => delete_storage!(values, I64),
        IntegerStorage::U8(values) => delete_storage!(values, U8),
        IntegerStorage::U16(values) => delete_storage!(values, U16),
        IntegerStorage::U32(values) => delete_storage!(values, U32),
        IntegerStorage::U64(values) => delete_storage!(values, U64),
    }
}

fn delete_tensor_linear(mut t: Tensor, idx: usize) -> Result<Value, RuntimeError> {
    let total = t.rows * t.cols;
    if idx == 0 || idx > total {
        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
    }
    if !(t.rows == 1 || t.cols == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    t.data.remove(idx - 1);
    if t.data.is_empty() {
        t.rows = 0;
        t.cols = 0;
        t.shape = vec![0, 0];
    } else if t.rows == 1 {
        t.cols = t.data.len();
        t.shape = vec![1, t.cols];
    } else {
        t.rows = t.data.len();
        t.shape = vec![t.rows, 1];
    }
    Ok(Value::Tensor(t))
}

fn delete_integer_tensor_linear(mut t: Tensor, idx: usize) -> Result<Value, RuntimeError> {
    let total = t.rows * t.cols;
    if idx == 0 || idx > total {
        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
    }
    if !(t.rows == 1 || t.cols == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }

    let storage = t
        .integer_data
        .take()
        .expect("integer deletion requires exact integer storage");
    let storage = delete_from_integer_storage(storage, idx - 1);
    let shape = deleted_vector_shape(t.rows, t.cols, storage.len());
    Tensor::new_integer(storage, shape)
        .map(Value::Tensor)
        .map_err(map_assignment_shape_error)
}

fn tensor_to_complex(t: Tensor) -> ComplexTensor {
    debug_assert!(t.integer_data.is_none());
    ComplexTensor {
        data: t.data.into_iter().map(|re| (re, 0.0)).collect(),
        integer_data: None,
        shape: t.shape,
        rows: t.rows,
        cols: t.cols,
    }
}

fn delete_complex_linear(mut t: ComplexTensor, idx: usize) -> Result<Value, RuntimeError> {
    let total = t.rows * t.cols;
    if idx == 0 || idx > total {
        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
    }
    if !(t.rows == 1 || t.cols == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    if let Some(storage) = t.integer_data.take() {
        let storage = delete_integer_complex_storage_positions(storage, &[idx - 1]);
        let shape = deleted_vector_shape(t.rows, t.cols, storage.len());
        return ComplexTensor::new_integer(storage, shape)
            .map(Value::ComplexTensor)
            .map_err(map_assignment_shape_error);
    }
    t.data.remove(idx - 1);
    if t.data.is_empty() {
        t.rows = 0;
        t.cols = 0;
        t.shape = vec![0, 0];
    } else if t.rows == 1 {
        t.cols = t.data.len();
        t.shape = vec![1, t.cols];
    } else {
        t.rows = t.data.len();
        t.shape = vec![t.rows, 1];
    }
    Ok(Value::ComplexTensor(t))
}

pub async fn rhs_to_real_scalar(rhs: &Value) -> Result<f64, RuntimeError> {
    match rhs {
        Value::Num(x) => Ok(*x),
        Value::Tensor(t2) => {
            if t2.data.len() == 1 {
                Ok(t2.data[0])
            } else {
                Err(mex("ScalarRequired", "RHS must be scalar"))
            }
        }
        Value::GpuTensor(h2) => {
            let total = h2.shape.iter().copied().product::<usize>();
            if total != 1 {
                return Err(mex("ScalarRequired", "RHS must be scalar"));
            }
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                mex(
                    "AccelerationProviderUnavailable",
                    "No acceleration provider registered",
                )
            })?;
            let host = provider
                .download(h2)
                .await
                .map_err(|e| map_acceleration_error("gather rhs", e))?;
            Ok(host.data[0])
        }
        _ => rhs
            .try_into()
            .map_err(|_| mex("NumericRequired", "RHS must be numeric")),
    }
}

pub async fn rhs_to_complex_scalar(rhs: &Value) -> Result<(f64, f64), RuntimeError> {
    match rhs {
        Value::Complex(re, im) => Ok((*re, *im)),
        Value::Num(n) => Ok((*n, 0.0)),
        Value::Int(i) => Ok((i.to_f64(), 0.0)),
        Value::Bool(b) => Ok((if *b { 1.0 } else { 0.0 }, 0.0)),
        Value::Tensor(t) if t.data.len() == 1 => Ok((t.data[0], 0.0)),
        Value::ComplexTensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::GpuTensor(h) => {
            let total = h.shape.iter().copied().product::<usize>();
            if total != 1 {
                return Err(mex("ScalarRequired", "RHS must be scalar"));
            }
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                mex(
                    "AccelerationProviderUnavailable",
                    "No acceleration provider registered",
                )
            })?;
            let host = provider
                .download(h)
                .await
                .map_err(|e| map_acceleration_error("gather rhs", e))?;
            Ok((host.data[0], 0.0))
        }
        _ => Err(mex("NumericRequired", "RHS must be numeric")),
    }
}

pub async fn assign_tensor_scalar(
    mut t: Tensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if indices.len() == 1 {
        let total = t.rows * t.cols;
        let idx = indices[0];
        if idx == 0 {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        if delete {
            if idx > total {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if !is_empty_tensor(rhs) {
                return Err(mex(
                    "DeletionRequiresEmptyRhs",
                    "Indexed deletion requires empty RHS",
                ));
            }
            return if t.integer_storage().is_some() {
                delete_integer_tensor_linear(t, idx)
            } else {
                delete_tensor_linear(t, idx)
            };
        }
        if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
            if t.integer_storage().is_some() {
                return Err(mex(
                    "UnsupportedTypedComplexInteger",
                    "typed complex integer assignment is not implemented",
                ));
            }
            return assign_complex_scalar(tensor_to_complex(t), indices, rhs, false).await;
        }
        if t.integer_storage().is_some() {
            return assign_integer_tensor_scalar(t, indices, rhs).await;
        }
        let val = rhs_to_real_scalar(rhs).await?;
        if idx > total {
            if !(t.rows == 1 || t.cols == 1) {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            let target_len = idx;
            if t.rows == 1 {
                t.data.resize(target_len, 0.0);
                t.cols = target_len;
                t.shape = vec![1, t.cols];
            } else {
                t.data.resize(target_len, 0.0);
                t.rows = target_len;
                t.shape = vec![t.rows, 1];
            }
        }
        t.data[idx - 1] = val;
        Ok(Value::Tensor(t))
    } else if indices.len() == 2 {
        let i = indices[0];
        let mut j = indices[1];
        let rows = t.rows;
        let cols = t.cols;
        if j == 0 {
            j = 1;
        }
        if j > cols {
            j = cols;
        }
        if i == 0 || i > rows {
            return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
        }
        if delete {
            return Err(mex(
                "UnsupportedDeletion",
                "Indexed deletion is only supported for linear vector indices",
            ));
        }
        if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
            if t.integer_storage().is_some() {
                return Err(mex(
                    "UnsupportedTypedComplexInteger",
                    "typed complex integer assignment is not implemented",
                ));
            }
            return assign_complex_scalar(tensor_to_complex(t), indices, rhs, false).await;
        }
        if t.integer_storage().is_some() {
            return assign_integer_tensor_scalar(t, indices, rhs).await;
        }
        let val = rhs_to_real_scalar(rhs).await?;
        let idx = (i - 1) + (j - 1) * rows;
        t.data[idx] = val;
        Ok(Value::Tensor(t))
    } else {
        Err(mex(
            "UnsupportedAssignmentRank",
            "Only 1D/2D scalar assignment supported",
        ))
    }
}

/// Assigns one sparse matrix element, growing scalar-indexed vectors and
/// matrices without densifying its CSC representation. Selector assignment
/// deliberately remains in the slice-assignment implementation.
pub async fn assign_sparse_scalar(
    sparse: SparseTensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if delete {
        if !is_empty_tensor(rhs) {
            return Err(mex(
                "DeletionRequiresEmptyRhs",
                "Indexed deletion requires empty RHS",
            ));
        }
        let updated = match indices {
            [index] => {
                let total = sparse
                    .rows
                    .checked_mul(sparse.cols)
                    .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?;
                if *index == 0 || *index > total {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                if sparse.rows == 1 {
                    sparse.with_deleted_columns(&[*index - 1])
                } else if sparse.cols == 1 {
                    sparse.with_deleted_rows(&[*index - 1])
                } else {
                    return Err(mex(
                        "UnsupportedDeletion",
                        "Linear sparse deletion is only supported for vectors",
                    ));
                }
            }
            [row, column] => {
                if *row == 0 || *row > sparse.rows || *column == 0 || *column > sparse.cols {
                    return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
                }
                if sparse.rows == 1 {
                    sparse.with_deleted_columns(&[*column - 1])
                } else if sparse.cols == 1 {
                    sparse.with_deleted_rows(&[*row - 1])
                } else {
                    return Err(mex(
                        "UnsupportedDeletion",
                        "Sparse deletion requires selecting complete rows or columns",
                    ));
                }
            }
            _ => {
                return Err(mex(
                    "UnsupportedDeletion",
                    "Sparse scalar deletion is only supported for vector indices",
                ))
            }
        }
        .map_err(map_assignment_shape_error)?;
        return Ok(Value::SparseTensor(updated));
    }
    let (sparse, row, col) = match indices {
        [index] => {
            let total = sparse
                .rows
                .checked_mul(sparse.cols)
                .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?;
            if *index == 0 {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if *index > total {
                let expanded = if sparse.rows == 1 || (sparse.rows == 0 && sparse.cols == 0) {
                    sparse.with_expanded_shape(1, *index)
                } else if sparse.cols == 1 {
                    sparse.with_expanded_shape(*index, 1)
                } else {
                    return Err(mex(
                        "IndexOutOfBounds",
                        "Linear sparse growth is only supported for vectors",
                    ));
                }
                .map_err(map_assignment_shape_error)?;
                if expanded.rows == 1 {
                    (expanded, 0, *index - 1)
                } else {
                    (expanded, *index - 1, 0)
                }
            } else {
                let rows = sparse.rows;
                (sparse, (*index - 1) % rows, (*index - 1) / rows)
            }
        }
        [row, column] => {
            if *row == 0 || *column == 0 {
                return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
            }
            let rows = sparse.rows.max(*row);
            let cols = sparse.cols.max(*column);
            let expanded = sparse
                .with_expanded_shape(rows, cols)
                .map_err(map_assignment_shape_error)?;
            (expanded, *row - 1, *column - 1)
        }
        _ => {
            return Err(mex(
                "UnsupportedAssignmentRank",
                "Only 1D/2D scalar assignment supported",
            ))
        }
    };

    let updated = if let Some(storage) = sparse.integer_storage() {
        let rhs = rhs_to_integer_assignment_scalar(rhs).await?;
        let value = integer_assignment::scalar_value(storage, &rhs);
        sparse
            .with_updated_integer_value(row, col, value)
            .map_err(map_assignment_shape_error)?
    } else {
        let value = rhs_to_real_scalar(rhs).await?;
        sparse
            .with_updated_value(row, col, value)
            .map_err(map_assignment_shape_error)?
    };
    Ok(Value::SparseTensor(updated))
}

async fn assign_integer_tensor_scalar(
    mut t: Tensor,
    indices: &[usize],
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let index = match indices {
        [index] => {
            let total = t.rows * t.cols;
            if *index == 0 {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if *index > total {
                if !(t.rows == 1 || t.cols == 1) {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                if t.rows == 1 {
                    t.cols = *index;
                    t.shape = vec![1, t.cols];
                } else {
                    t.rows = *index;
                    t.shape = vec![t.rows, 1];
                }
            }
            *index - 1
        }
        [row, column] => {
            let rows = t.rows;
            let cols = t.cols;
            let column = if *column == 0 { 1 } else { (*column).min(cols) };
            if *row == 0 || *row > rows {
                return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
            }
            *row - 1 + (column - 1) * rows
        }
        _ => {
            return Err(mex(
                "UnsupportedAssignmentRank",
                "Only 1D/2D scalar assignment supported",
            ))
        }
    };

    let rhs = rhs_to_integer_assignment_scalar(rhs).await?;
    let storage = t
        .integer_data
        .take()
        .expect("integer assignment requires exact integer storage");
    let storage = assign_integer_storage(storage, index, t.rows * t.cols, &rhs);
    Tensor::new_integer(storage, t.shape)
        .map(Value::Tensor)
        .map_err(map_assignment_shape_error)
}

pub async fn assign_complex_scalar(
    mut t: ComplexTensor,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if !delete && t.integer_data.is_some() {
        return Err(mex(
            "UnsupportedTypedComplexInteger",
            "typed complex integer assignment is not implemented",
        ));
    }
    if indices.len() == 1 {
        let total = t.rows * t.cols;
        let idx = indices[0];
        if idx == 0 {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        if delete {
            if idx > total {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if !is_empty_tensor(rhs) {
                return Err(mex(
                    "DeletionRequiresEmptyRhs",
                    "Indexed deletion requires empty RHS",
                ));
            }
            return delete_complex_linear(t, idx);
        }
        let val = rhs_to_complex_scalar(rhs).await?;
        if idx > total {
            if !(t.rows == 1 || t.cols == 1) {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            let target_len = idx;
            if t.rows == 1 {
                t.data.resize(target_len, (0.0, 0.0));
                t.cols = target_len;
                t.shape = vec![1, t.cols];
            } else {
                t.data.resize(target_len, (0.0, 0.0));
                t.rows = target_len;
                t.shape = vec![t.rows, 1];
            }
        }
        t.data[idx - 1] = val;
        Ok(Value::ComplexTensor(t))
    } else if indices.len() == 2 {
        let i = indices[0];
        let mut j = indices[1];
        let rows = t.rows;
        let cols = t.cols;
        if j == 0 {
            j = 1;
        }
        if j > cols {
            j = cols;
        }
        if i == 0 || i > rows {
            return Err(mex("SubscriptOutOfBounds", "Subscript out of bounds"));
        }
        if delete {
            return Err(mex(
                "UnsupportedDeletion",
                "Indexed deletion is only supported for linear vector indices",
            ));
        }
        let val = rhs_to_complex_scalar(rhs).await?;
        let idx = (i - 1) + (j - 1) * rows;
        t.data[idx] = val;
        Ok(Value::ComplexTensor(t))
    } else {
        Err(mex(
            "UnsupportedAssignmentRank",
            "Only 1D/2D scalar assignment supported",
        ))
    }
}

pub async fn assign_gpu_scalar(
    h: &runmat_accelerate_api::GpuTensorHandle,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered",
        )
    })?;
    let host = provider
        .download(h)
        .await
        .map_err(|e| map_acceleration_error("gather for assignment", e))?;
    let t = Tensor::new(host.data, host.shape).map_err(map_assignment_shape_error)?;
    let Value::Tensor(updated) = assign_tensor_scalar(t, indices, rhs, delete).await? else {
        unreachable!()
    };
    let view = runmat_accelerate_api::HostTensorView {
        data: &updated.data,
        shape: &updated.shape,
    };
    let new_h = provider
        .upload(&view)
        .map_err(|e| map_acceleration_error("reupload after assignment", e))?;
    Ok(Value::GpuTensor(new_h))
}

#[cfg(test)]
mod tests {
    use super::{
        assign_sparse_scalar, assign_tensor_scalar, map_acceleration_error,
        map_assignment_shape_error,
    };
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor, Value};

    #[test]
    fn integer_linear_assignment_preserves_exact_uint64_rhs() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![1, 2]).expect("tensor");
        let result = block_on(assign_tensor_scalar(
            tensor,
            &[2],
            &Value::Int(IntValue::U64(u64::MAX)),
            false,
        ))
        .expect("assignment");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, u64::MAX]))
        );
    }

    #[test]
    fn integer_subscript_assignment_rounds_and_saturates() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I8(vec![0, 0, 0, 0]), vec![2, 2]).expect("tensor");
        let rounded = block_on(assign_tensor_scalar(
            tensor,
            &[2, 1],
            &Value::Num(3.5),
            false,
        ))
        .expect("assignment");
        let Value::Tensor(rounded) = rounded else {
            panic!("expected tensor");
        };
        let saturated = block_on(assign_tensor_scalar(
            rounded,
            &[1, 2],
            &Value::Int(IntValue::U64(u64::MAX)),
            false,
        ))
        .expect("assignment");

        let Value::Tensor(output) = saturated else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![0, 4, i8::MAX, 0]))
        );
    }

    #[test]
    fn integer_vector_growth_and_deletion_preserve_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![7]), vec![1, 1]).expect("tensor");
        let grown = block_on(assign_tensor_scalar(
            tensor,
            &[3],
            &Value::Int(IntValue::I16(9)),
            false,
        ))
        .expect("growth");
        let Value::Tensor(grown) = grown else {
            panic!("expected tensor");
        };
        assert_eq!(
            grown.integer_storage(),
            Some(&IntegerStorage::I16(vec![7, 0, 9]))
        );

        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let deleted = block_on(assign_tensor_scalar(grown, &[2], &empty, true)).expect("delete");
        let Value::Tensor(output) = deleted else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I16(vec![7, 9]))
        );
        assert_eq!(output.shape, vec![1, 2]);
    }

    #[test]
    fn sparse_integer_assignment_preserves_exact_uint64_and_elides_zero() {
        let sparse = runmat_builtins::SparseTensor::new_integer(
            2,
            2,
            vec![0, 0, 0],
            vec![],
            IntegerStorage::U64(vec![]),
        )
        .expect("sparse");
        let Value::SparseTensor(inserted) = block_on(assign_sparse_scalar(
            sparse,
            &[2, 2],
            &Value::Int(IntValue::U64(u64::MAX)),
            false,
        ))
        .expect("insert") else {
            panic!("expected sparse output");
        };
        assert_eq!(inserted.row_indices, vec![1]);
        assert_eq!(inserted.col_ptrs, vec![0, 0, 1]);
        assert_eq!(
            inserted.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX]))
        );

        let Value::SparseTensor(removed) = block_on(assign_sparse_scalar(
            inserted,
            &[4],
            &Value::Num(0.0),
            false,
        ))
        .expect("remove") else {
            panic!("expected sparse output");
        };
        assert_eq!(removed.col_ptrs, vec![0, 0, 0]);
        assert_eq!(
            removed.integer_storage(),
            Some(&IntegerStorage::U64(vec![]))
        );
    }

    #[test]
    fn sparse_integer_assignment_preserves_every_integer_class() {
        let cases = vec![
            (IntegerStorage::I8(vec![]), IntValue::I8(i8::MIN)),
            (IntegerStorage::I16(vec![]), IntValue::I16(i16::MIN)),
            (IntegerStorage::I32(vec![]), IntValue::I32(i32::MIN)),
            (IntegerStorage::I64(vec![]), IntValue::I64(i64::MIN)),
            (IntegerStorage::U8(vec![]), IntValue::U8(u8::MAX)),
            (IntegerStorage::U16(vec![]), IntValue::U16(u16::MAX)),
            (IntegerStorage::U32(vec![]), IntValue::U32(u32::MAX)),
            (IntegerStorage::U64(vec![]), IntValue::U64(u64::MAX)),
        ];

        for (storage, value) in cases {
            let sparse =
                runmat_builtins::SparseTensor::new_integer(1, 1, vec![0, 0], vec![], storage)
                    .expect("sparse");
            let Value::SparseTensor(updated) = block_on(assign_sparse_scalar(
                sparse,
                &[1, 1],
                &Value::Int(value.clone()),
                false,
            ))
            .expect("assignment") else {
                panic!("expected sparse output");
            };
            assert_eq!(updated.integer_at(0, 0), Some(value));
        }
    }

    #[test]
    fn sparse_integer_assignment_rounds_and_saturates_numeric_rhs() {
        let sparse = runmat_builtins::SparseTensor::new_integer(
            1,
            2,
            vec![0, 0, 0],
            vec![],
            IntegerStorage::I8(vec![]),
        )
        .expect("sparse");
        let Value::SparseTensor(rounded) = block_on(assign_sparse_scalar(
            sparse,
            &[1, 1],
            &Value::Num(3.5),
            false,
        ))
        .expect("round") else {
            panic!("expected sparse output");
        };
        let Value::SparseTensor(saturated) = block_on(assign_sparse_scalar(
            rounded,
            &[1, 2],
            &Value::Int(IntValue::U64(u64::MAX)),
            false,
        ))
        .expect("saturate") else {
            panic!("expected sparse output");
        };
        assert_eq!(
            saturated.integer_storage(),
            Some(&IntegerStorage::I8(vec![4, i8::MAX]))
        );
    }

    #[test]
    fn sparse_scalar_assignment_reports_stable_negative_path_errors() {
        let sparse = runmat_builtins::SparseTensor::zeros(1, 1);
        let out_of_bounds = block_on(assign_sparse_scalar(
            sparse.clone(),
            &[0],
            &Value::Num(1.0),
            false,
        ))
        .expect_err("out-of-bounds assignment must fail");
        assert_eq!(out_of_bounds.identifier(), Some("RunMat:IndexOutOfBounds"));

        let subscript_out_of_bounds = block_on(assign_sparse_scalar(
            sparse.clone(),
            &[1, 0],
            &Value::Num(1.0),
            false,
        ))
        .expect_err("invalid sparse subscript must fail");
        assert_eq!(
            subscript_out_of_bounds.identifier(),
            Some("RunMat:SubscriptOutOfBounds")
        );

        let deletion = block_on(assign_sparse_scalar(
            runmat_builtins::SparseTensor::zeros(2, 2),
            &[1],
            &Value::Tensor(Tensor::zeros(vec![0, 0])),
            true,
        ))
        .expect_err("matrix linear deletion must remain unsupported");
        assert_eq!(deletion.identifier(), Some("RunMat:UnsupportedDeletion"));

        let nonempty_deletion = block_on(assign_sparse_scalar(
            sparse.clone(),
            &[1],
            &Value::Num(1.0),
            true,
        ))
        .expect_err("deletion with a nonempty RHS must fail");
        assert_eq!(
            nonempty_deletion.identifier(),
            Some("RunMat:DeletionRequiresEmptyRhs")
        );

        let rank = block_on(assign_sparse_scalar(
            sparse,
            &[1, 1, 1],
            &Value::Num(1.0),
            false,
        ))
        .expect_err("rank-three scalar assignment must fail");
        assert_eq!(rank.identifier(), Some("RunMat:UnsupportedAssignmentRank"));
    }

    #[test]
    fn assignment_shape_error_mapping_reports_identifier() {
        let err = map_assignment_shape_error("invalid shape");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn assignment_acceleration_error_mapping_reports_identifier() {
        let err = map_acceleration_error("gather rhs", "provider failed");
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
    }
}
