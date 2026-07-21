use crate::indexing::write_slice::deleted_vector_shape;
use crate::interpreter::errors::mex;
use runmat_builtins::{ComplexTensor, IntValue, IntegerStorage, Tensor, Value};
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

enum IntegerAssignmentRhs {
    Exact(IntValue),
    Float(f64),
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
) -> Result<IntegerAssignmentRhs, RuntimeError> {
    match rhs {
        Value::Int(value) => Ok(IntegerAssignmentRhs::Exact(value.clone())),
        Value::Num(value) => Ok(IntegerAssignmentRhs::Float(*value)),
        Value::Bool(value) => Ok(IntegerAssignmentRhs::Float(if *value { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) if tensor.data.len() == 1 => match tensor.integer_storage() {
            Some(storage) => Ok(IntegerAssignmentRhs::Exact(integer_storage_scalar(storage))),
            None => Ok(IntegerAssignmentRhs::Float(tensor.data[0])),
        },
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(IntegerAssignmentRhs::Float(if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }))
        }
        _ => rhs_to_real_scalar(rhs)
            .await
            .map(IntegerAssignmentRhs::Float),
    }
}

fn cast_signed_assignment(value: &IntegerAssignmentRhs, min: i64, max: i64) -> i64 {
    match value {
        IntegerAssignmentRhs::Exact(value) => value.to_i64().clamp(min, max),
        IntegerAssignmentRhs::Float(value) if value.is_nan() => 0,
        IntegerAssignmentRhs::Float(value) if value.is_infinite() => {
            if value.is_sign_negative() {
                min
            } else {
                max
            }
        }
        IntegerAssignmentRhs::Float(value) => value.round().clamp(min as f64, max as f64) as i64,
    }
}

fn unsigned_exact_value(value: &IntValue) -> u64 {
    match value {
        IntValue::U64(value) => *value,
        _ => value.to_i64().max(0) as u64,
    }
}

fn cast_unsigned_assignment(value: &IntegerAssignmentRhs, max: u64) -> u64 {
    match value {
        IntegerAssignmentRhs::Exact(value) => unsigned_exact_value(value).min(max),
        IntegerAssignmentRhs::Float(value) if value.is_nan() || value.is_sign_negative() => 0,
        IntegerAssignmentRhs::Float(value) if value.is_infinite() => max,
        IntegerAssignmentRhs::Float(value) => value.round().clamp(0.0, max as f64) as u64,
    }
}

fn assign_integer_storage(
    storage: IntegerStorage,
    index: usize,
    target_len: usize,
    rhs: &IntegerAssignmentRhs,
) -> IntegerStorage {
    macro_rules! assign_storage {
        ($values:expr, $variant:ident, $zero:expr, $value:expr) => {{
            let mut values = $values;
            values.resize(target_len, $zero);
            values[index] = $value;
            IntegerStorage::$variant(values)
        }};
    }

    match storage {
        IntegerStorage::I8(values) => assign_storage!(
            values,
            I8,
            0,
            cast_signed_assignment(rhs, i8::MIN as i64, i8::MAX as i64) as i8
        ),
        IntegerStorage::I16(values) => assign_storage!(
            values,
            I16,
            0,
            cast_signed_assignment(rhs, i16::MIN as i64, i16::MAX as i64) as i16
        ),
        IntegerStorage::I32(values) => assign_storage!(
            values,
            I32,
            0,
            cast_signed_assignment(rhs, i32::MIN as i64, i32::MAX as i64) as i32
        ),
        IntegerStorage::I64(values) => assign_storage!(
            values,
            I64,
            0,
            cast_signed_assignment(rhs, i64::MIN, i64::MAX)
        ),
        IntegerStorage::U8(values) => {
            assign_storage!(
                values,
                U8,
                0,
                cast_unsigned_assignment(rhs, u8::MAX as u64) as u8
            )
        }
        IntegerStorage::U16(values) => assign_storage!(
            values,
            U16,
            0,
            cast_unsigned_assignment(rhs, u16::MAX as u64) as u16
        ),
        IntegerStorage::U32(values) => assign_storage!(
            values,
            U32,
            0,
            cast_unsigned_assignment(rhs, u32::MAX as u64) as u32
        ),
        IntegerStorage::U64(values) => {
            assign_storage!(values, U64, 0, cast_unsigned_assignment(rhs, u64::MAX))
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
    ComplexTensor {
        data: t.data.into_iter().map(|re| (re, 0.0)).collect(),
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
    use super::{assign_tensor_scalar, map_acceleration_error, map_assignment_shape_error};
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
