use crate::indexing::integer_assignment::{
    self, ComplexIntegerAssignmentValue, IntegerAssignmentValue,
};
use crate::indexing::write_slice::{
    delete_integer_complex_storage_positions, deleted_vector_shape, download_integer_tensor,
    real_tensor_to_complex, upload_tensor_to_gpu,
};
use crate::interpreter::errors::mex;
use runmat_builtins::{
    ComplexTensor, IntValue, IntegerStorage, NumericDType, NumericScalar, NumericStorage,
    SparseTensor, Tensor, Value,
};
use runmat_runtime::builtins::common::tensor::{
    complex_tensor_element_len, complex_tensor_value_complex64, is_scalar_tensor,
    tensor_element_len, tensor_value_f64,
};
use runmat_runtime::RuntimeError;

fn map_assignment_shape_error(err: impl std::fmt::Display) -> RuntimeError {
    mex("ShapeMismatch", &format!("assignment: {err}"))
}

fn map_acceleration_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    mex("AccelerationOperationFailed", &format!("{context}: {err}"))
}

fn is_empty_tensor(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if tensor_element_len(t) == 0 || t.rows == 0 || t.cols == 0)
        || matches!(value, Value::ComplexTensor(t) if complex_tensor_element_len(t) == 0 || t.rows == 0 || t.cols == 0)
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
        Value::Tensor(tensor) if is_scalar_tensor(tensor) => {
            let value = tensor
                .numeric_value_at(0)
                .expect("scalar tensor must contain one numeric value");
            Ok(match value.into_int_value() {
                Some(value) => IntegerAssignmentValue::Exact(value),
                None => IntegerAssignmentValue::Float(value.materialize_f64()),
            })
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(IntegerAssignmentValue::Float(if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }))
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_some() =>
        {
            let tensor = download_integer_tensor(handle).await?;
            let storage = tensor
                .integer_storage()
                .expect("exact integer GPU gather must retain integer storage");
            if storage.len() != 1 {
                return Err(mex(
                    "InvalidAssignmentRhs",
                    "integer gpuArray assignment rhs must be scalar",
                ));
            }
            Ok(IntegerAssignmentValue::Exact(integer_storage_scalar(
                storage,
            )))
        }
        _ => rhs_to_real_scalar(rhs)
            .await
            .map(IntegerAssignmentValue::Float),
    }
}

async fn rhs_to_complex_integer_assignment_scalar(
    rhs: &Value,
) -> Result<ComplexIntegerAssignmentValue, RuntimeError> {
    let scalar = |real, imag| ComplexIntegerAssignmentValue {
        real: IntegerAssignmentValue::Float(real),
        imag: IntegerAssignmentValue::Float(imag),
    };
    match rhs {
        Value::Complex(real, imag) => Ok(scalar(*real, *imag)),
        Value::ComplexTensor(tensor) if complex_tensor_element_len(tensor) == 1 => {
            if let Some(storage) = &tensor.integer_storage() {
                return Ok(ComplexIntegerAssignmentValue {
                    real: IntegerAssignmentValue::Exact(
                        storage.real.value_at(0).expect("scalar component"),
                    ),
                    imag: IntegerAssignmentValue::Exact(
                        storage.imag.value_at(0).expect("scalar component"),
                    ),
                });
            }
            let (real, imag) = tensor.materialize_f64()[0];
            Ok(scalar(real, imag))
        }
        _ => Ok(ComplexIntegerAssignmentValue {
            real: rhs_to_integer_assignment_scalar(rhs).await?,
            imag: IntegerAssignmentValue::Float(0.0),
        }),
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

fn delete_tensor_linear(t: Tensor, idx: usize) -> Result<Value, RuntimeError> {
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
    let rows = t.rows;
    let cols = t.cols;
    let mut storage = t
        .into_numeric_storage()
        .map_err(map_assignment_shape_error)?;
    storage
        .remove_positions(&[idx - 1])
        .map_err(map_assignment_shape_error)?;
    let shape = deleted_vector_shape(rows, cols, storage.len());
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(map_assignment_shape_error)
}

fn delete_complex_linear(t: ComplexTensor, idx: usize) -> Result<Value, RuntimeError> {
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
    if let Some(storage) = t.integer_storage().cloned() {
        let storage = delete_integer_complex_storage_positions(storage, &[idx - 1]);
        let shape = deleted_vector_shape(t.rows, t.cols, storage.len());
        return ComplexTensor::new_integer(storage, shape)
            .map(Value::ComplexTensor)
            .map_err(map_assignment_shape_error);
    }
    let dtype = t.numeric_dtype();
    let rows = t.rows;
    let cols = t.cols;
    let mut values = t.materialize_f64();
    values.remove(idx - 1);
    let shape = deleted_vector_shape(rows, cols, values.len());
    ComplexTensor::from_f64_values_with_dtype(values, shape, dtype)
        .map(Value::ComplexTensor)
        .map_err(map_assignment_shape_error)
}

pub async fn rhs_to_real_scalar(rhs: &Value) -> Result<f64, RuntimeError> {
    match rhs {
        Value::Num(x) => Ok(*x),
        Value::Tensor(t2) => {
            if is_scalar_tensor(t2) {
                Ok(tensor_value_f64(t2, 0))
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
        Value::Tensor(t) if is_scalar_tensor(t) => Ok((tensor_value_f64(t, 0), 0.0)),
        Value::ComplexTensor(t) if complex_tensor_element_len(t) == 1 => {
            let value = complex_tensor_value_complex64(t, 0);
            Ok((value.re, value.im))
        }
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
    t: Tensor,
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
            return delete_tensor_linear(t, idx);
        }
        if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
            let tensor = real_tensor_to_complex(t, "scalar complex promotion")?;
            return assign_complex_scalar(tensor, indices, rhs, false).await;
        }
        let shape = if idx > total {
            if !(t.rows == 1 || t.cols == 1) {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            if t.rows == 1 {
                vec![1, idx]
            } else {
                vec![idx, 1]
            }
        } else {
            t.shape.clone()
        };
        assign_real_tensor_scalar(t, idx - 1, shape, rhs).await
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
            let tensor = real_tensor_to_complex(t, "scalar complex promotion")?;
            return assign_complex_scalar(tensor, indices, rhs, false).await;
        }
        let idx = (i - 1) + (j - 1) * rows;
        let shape = t.shape.clone();
        assign_real_tensor_scalar(t, idx, shape, rhs).await
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
    } else if sparse.numeric_dtype() == NumericDType::F32 {
        let value = rhs_to_real_scalar(rhs).await? as f32;
        sparse
            .with_updated_f32_value(row, col, value)
            .map_err(map_assignment_shape_error)?
    } else {
        let value = rhs_to_real_scalar(rhs).await?;
        sparse
            .with_updated_value(row, col, value)
            .map_err(map_assignment_shape_error)?
    };
    Ok(Value::SparseTensor(updated))
}

async fn assign_real_tensor_scalar(
    t: Tensor,
    index: usize,
    shape: Vec<usize>,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let mut storage = t
        .into_numeric_storage()
        .map_err(map_assignment_shape_error)?;
    let target_len = shape.iter().product();
    storage.resize_zeroed(target_len);
    storage = match storage.into_integer_storage() {
        Ok(storage) => {
            let rhs = rhs_to_integer_assignment_scalar(rhs).await?;
            NumericStorage::from_integer_storage(assign_integer_storage(
                storage, index, target_len, &rhs,
            ))
        }
        Err(mut storage) => {
            let rhs = rhs_to_real_scalar(rhs).await?;
            let value = match storage.numeric_dtype() {
                NumericDType::F64 => NumericScalar::F64(rhs),
                NumericDType::F32 => NumericScalar::F32(rhs as f32),
                _ => unreachable!("non-integer storage must be floating"),
            };
            storage
                .set_value(index, value)
                .map_err(map_assignment_shape_error)?;
            storage
        }
    };
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(map_assignment_shape_error)
}

pub async fn assign_complex_scalar(
    mut t: ComplexTensor,
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
            return delete_complex_linear(t, idx);
        }
        if t.integer_storage().is_some() {
            return assign_typed_complex_integer_scalar(t, idx, rhs).await;
        }
        let val = rhs_to_complex_scalar(rhs).await?;
        let dtype = t.numeric_dtype();
        let mut values = t.materialize_f64();
        if idx > total {
            if !(t.rows == 1 || t.cols == 1) {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            let target_len = idx;
            if t.rows == 1 {
                values.resize(target_len, (0.0, 0.0));
                t.cols = target_len;
                t.shape = vec![1, t.cols];
            } else {
                values.resize(target_len, (0.0, 0.0));
                t.rows = target_len;
                t.shape = vec![t.rows, 1];
            }
        }
        values[idx - 1] = val;
        ComplexTensor::from_f64_values_with_dtype(values, t.shape, dtype)
            .map(Value::ComplexTensor)
            .map_err(map_assignment_shape_error)
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
        let idx = (i - 1) + (j - 1) * rows;
        if t.integer_storage().is_some() {
            return assign_typed_complex_integer_scalar(t, idx + 1, rhs).await;
        }
        let val = rhs_to_complex_scalar(rhs).await?;
        t.set_f64_assignment_at(idx, val.0, val.1)
            .map_err(map_assignment_shape_error)?;
        Ok(Value::ComplexTensor(t))
    } else {
        Err(mex(
            "UnsupportedAssignmentRank",
            "Only 1D/2D scalar assignment supported",
        ))
    }
}

async fn assign_typed_complex_integer_scalar(
    mut tensor: ComplexTensor,
    index: usize,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let total = tensor.rows * tensor.cols;
    if index == 0 {
        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
    }
    if index > total {
        if !(tensor.rows == 1 || tensor.cols == 1) {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        if tensor.rows == 1 {
            tensor.cols = index;
            tensor.shape = vec![1, tensor.cols];
        } else {
            tensor.rows = index;
            tensor.shape = vec![tensor.rows, 1];
        }
    }
    let rhs = rhs_to_complex_integer_assignment_scalar(rhs).await?;
    let storage = tensor
        .integer_storage()
        .cloned()
        .expect("typed complex assignment requires exact storage");
    let real = assign_integer_storage(
        storage.real,
        index - 1,
        tensor.rows * tensor.cols,
        &rhs.real,
    );
    let imag = assign_integer_storage(
        storage.imag,
        index - 1,
        tensor.rows * tensor.cols,
        &rhs.imag,
    );
    runmat_builtins::IntegerComplexStorage::new(real, imag)
        .and_then(|storage| ComplexTensor::new_integer(storage, tensor.shape))
        .map(Value::ComplexTensor)
        .map_err(map_assignment_shape_error)
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
    if runmat_accelerate_api::handle_integer_type(h).is_some() {
        let tensor = download_integer_tensor(h).await?;
        let Value::Tensor(updated) = assign_tensor_scalar(tensor, indices, rhs, delete).await?
        else {
            unreachable!("real integer scalar assignment must produce a real tensor")
        };
        return upload_tensor_to_gpu(&updated);
    }
    let host = provider
        .download(h)
        .await
        .map_err(|e| map_acceleration_error("gather for assignment", e))?;
    let t = Tensor::new(host.data, host.shape).map_err(map_assignment_shape_error)?;
    let Value::Tensor(updated) = assign_tensor_scalar(t, indices, rhs, delete).await? else {
        unreachable!()
    };
    let data = updated.materialize_f64();
    let view = runmat_accelerate_api::HostTensorView {
        data: &data,
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
        assign_complex_scalar, assign_sparse_scalar, assign_tensor_scalar, map_acceleration_error,
        map_assignment_shape_error,
    };
    use futures::executor::block_on;
    use runmat_builtins::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, NumericDType,
        NumericStorage, Tensor, Value,
    };

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
    fn native_single_linear_growth_assignment_and_deletion_preserve_class() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).expect("single tensor");
        let Value::Tensor(grown) = block_on(assign_tensor_scalar(
            tensor,
            &[4],
            &Value::Num(1.234_567_890_123),
            false,
        ))
        .expect("single growth assignment") else {
            panic!("expected tensor");
        };
        assert_eq!(grown.numeric_dtype(), NumericDType::F32);
        assert_eq!(grown.shape, vec![1, 4]);
        assert_eq!(
            grown.clone().into_numeric_storage(),
            Ok(NumericStorage::F32(vec![
                1.0,
                2.0,
                0.0,
                1.234_567_890_123_f64 as f32,
            ]))
        );

        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let Value::Tensor(deleted) =
            block_on(assign_tensor_scalar(grown, &[2], &empty, true)).expect("single deletion")
        else {
            panic!("expected tensor");
        };
        assert_eq!(deleted.numeric_dtype(), NumericDType::F32);
        assert_eq!(deleted.shape, vec![1, 3]);
        assert_eq!(
            deleted.into_numeric_storage(),
            Ok(NumericStorage::F32(vec![
                1.0,
                0.0,
                1.234_567_890_123_f64 as f32,
            ]))
        );
    }

    #[test]
    fn native_complex_single_linear_growth_assignment_and_deletion_preserve_class() {
        let tensor = ComplexTensor::from_f32(vec![(1.0, -1.0), (2.0, -2.0)], vec![1, 2]).unwrap();
        let Value::ComplexTensor(grown) = block_on(assign_complex_scalar(
            tensor,
            &[4],
            &Value::Complex(1.234_567_890_123, -9.876_543_210_987),
            false,
        ))
        .expect("complex single growth assignment") else {
            panic!("expected complex tensor");
        };
        assert_eq!(grown.numeric_dtype(), NumericDType::F32);
        assert_eq!(grown.shape, vec![1, 4]);
        assert_eq!(
            grown.as_f32_slice(),
            Some(
                &[
                    (1.0, -1.0),
                    (2.0, -2.0),
                    (0.0, 0.0),
                    (1.234_567_890_123_f64 as f32, -9.876_543_210_987_f64 as f32,),
                ][..]
            )
        );

        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let Value::ComplexTensor(deleted) =
            block_on(assign_complex_scalar(grown, &[2], &empty, true))
                .expect("complex single deletion")
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(deleted.numeric_dtype(), NumericDType::F32);
        assert_eq!(deleted.shape, vec![1, 3]);
        assert_eq!(
            deleted.as_f32_slice(),
            Some(
                &[
                    (1.0, -1.0),
                    (0.0, 0.0),
                    (1.234_567_890_123_f64 as f32, -9.876_543_210_987_f64 as f32,),
                ][..]
            )
        );
    }

    #[test]
    fn integer_scalar_complex_assignment_promotes_without_losing_wide_components() {
        let tensor = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![1, 2])
            .expect("tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![i64::MAX]),
                    IntegerStorage::I64(vec![i64::MIN]),
                )
                .expect("integer complex storage"),
                vec![1, 1],
            )
            .expect("rhs"),
        );

        let result = block_on(assign_tensor_scalar(tensor, &[2], &rhs, false))
            .expect("typed complex integer assignment");

        let Value::ComplexTensor(output) = result else {
            panic!("integer tensor should promote to typed complex integer tensor");
        };
        assert_eq!(
            output
                .integer_storage()
                .as_ref()
                .map(|storage| (&storage.real, &storage.imag)),
            Some((
                &IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                &IntegerStorage::I64(vec![0, i64::MIN]),
            ))
        );
    }

    #[test]
    fn linear_assignment_reads_typed_integer_rhs_without_mirror() {
        let tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).expect("tensor");
        let rhs = Tensor::new_integer(IntegerStorage::U16(vec![11]), vec![1, 1]).expect("rhs");

        let result = block_on(assign_tensor_scalar(
            tensor,
            &[2],
            &Value::Tensor(rhs),
            false,
        ))
        .expect("assignment");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.materialize_f64(), vec![0.0, 11.0]);
    }

    #[test]
    fn integer_linear_assignment_reads_typed_integer_rhs_without_mirror() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).expect("tensor");
        let rhs = Tensor::new_integer(IntegerStorage::I16(vec![7]), vec![1, 1]).expect("rhs");

        let result = block_on(assign_tensor_scalar(
            tensor,
            &[2],
            &Value::Tensor(rhs),
            false,
        ))
        .expect("assignment");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I16(vec![1, 7]))
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
    fn sparse_single_scalar_assignment_preserves_class_and_growth() {
        let sparse = runmat_builtins::SparseTensor::zeros_f32(1, 1);
        let Value::SparseTensor(updated) = block_on(assign_sparse_scalar(
            sparse,
            &[2, 2],
            &Value::Num(1.0 / 3.0),
            false,
        ))
        .expect("single sparse assignment") else {
            panic!("expected sparse output");
        };
        assert_eq!(updated.numeric_dtype(), NumericDType::F32);
        assert_eq!(updated.shape(), vec![2, 2]);
        assert_eq!(updated.col_ptrs, vec![0, 0, 1]);
        assert_eq!(updated.row_indices, vec![1]);
        assert_eq!(updated.as_f32_slice(), Some(&[(1.0 / 3.0) as f32][..]));
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
