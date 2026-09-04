use runmat_builtins::{REALSQRT_ERROR_DOMAIN, REALSQRT_ERROR_INVALID_INPUT};
use runmat_value::{NumericDType, NumericStorage, SparseTensor, Tensor, Value};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::errors;

pub(super) fn evaluate(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(super::BUILTIN_NAME, value)
        .map_err(|detail| errors::with_detail(&REALSQRT_ERROR_INVALID_INPUT, detail))?;
    evaluate_tensor(tensor)
}

pub(super) fn evaluate_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(errors::internal)?;
    let output = match storage {
        NumericStorage::F64(values) => {
            ensure_nonnegative(&values)?;
            NumericStorage::F64(
                values
                    .into_iter()
                    .map(|value| canonical_zero(value.sqrt()))
                    .collect(),
            )
        }
        NumericStorage::F32(values) => {
            ensure_nonnegative(&values)?;
            NumericStorage::F32(
                values
                    .into_iter()
                    .map(|value| canonical_zero(value.sqrt()))
                    .collect(),
            )
        }
        _ => {
            return Err(errors::with_detail(
                &REALSQRT_ERROR_INVALID_INPUT,
                "expected real single or double input",
            ))
        }
    };
    let output = Tensor::from_numeric_storage(output, shape).map_err(errors::internal)?;
    Ok(tensor_into_value(output))
}

pub(super) fn evaluate_sparse(sparse: SparseTensor) -> BuiltinResult<Value> {
    if sparse.integer_storage().is_some() || sparse.is_logical() || sparse.is_complex() {
        return Err(errors::with_detail(
            &REALSQRT_ERROR_INVALID_INPUT,
            "expected real single or double input",
        ));
    }
    let dtype = sparse.numeric_dtype();
    let values = sparse.materialize_f64();
    ensure_nonnegative(&values)?;
    let values = values.into_iter().map(f64::sqrt).collect::<Vec<_>>();
    let output = match dtype {
        Some(NumericDType::F32) => SparseTensor::new_f32(
            sparse.rows,
            sparse.cols,
            sparse.col_ptrs,
            sparse.row_indices,
            values.into_iter().map(|value| value as f32).collect(),
        ),
        Some(NumericDType::F64) => SparseTensor::new(
            sparse.rows,
            sparse.cols,
            sparse.col_ptrs,
            sparse.row_indices,
            values,
        ),
        Some(_) | None => unreachable!("non-floating sparse input rejected above"),
    };
    output.map(Value::SparseTensor).map_err(errors::internal)
}

fn ensure_nonnegative<T>(values: &[T]) -> BuiltinResult<()>
where
    T: Copy + PartialOrd + From<u8>,
{
    if values.iter().any(|value| *value < T::from(0)) {
        return Err(errors::with_detail(
            &REALSQRT_ERROR_DOMAIN,
            "input contains negative values",
        ));
    }
    Ok(())
}

fn canonical_zero<T>(value: T) -> T
where
    T: Copy + PartialEq + From<u8>,
{
    if value == T::from(0) {
        T::from(0)
    } else {
        value
    }
}

fn tensor_into_value(tensor: Tensor) -> Value {
    if tensor.numeric_dtype() == NumericDType::F64 {
        tensor::tensor_into_value(tensor)
    } else {
        Value::Tensor(tensor)
    }
}
