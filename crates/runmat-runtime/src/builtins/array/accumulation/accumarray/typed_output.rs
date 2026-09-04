use runmat_value::{
    CharArray, IntValue, IntegerStorage, LogicalArray, NumericDType, Tensor, Value,
};

use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

use super::{data, error, sparse};

pub(super) fn characters(
    values: Vec<Value>,
    shape: Vec<usize>,
    is_sparse: bool,
) -> BuiltinResult<Value> {
    require_dense(is_sparse)?;
    let characters = values
        .into_iter()
        .map(|value| match value {
            Value::CharArray(chars) => chars.data.first().copied().ok_or_else(class_mismatch),
            _ => Err(class_mismatch()),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    CharArray::from_column_major(characters, shape)
        .map(Value::CharArray)
        .map_err(error::invalid)
}

pub(super) fn integers(
    values: Vec<IntValue>,
    shape: Vec<usize>,
    is_sparse: bool,
) -> BuiltinResult<Value> {
    require_dense(is_sparse)?;
    let prototype = values
        .first()
        .cloned()
        .ok_or_else(|| error::invalid("accumarray: empty integer result"))?;
    let storage = IntegerStorage::from_scalar(prototype)
        .from_exact_values_like(values)
        .map_err(|_| class_mismatch())?;
    Tensor::new_integer(storage, shape)
        .map(Value::Tensor)
        .map_err(error::invalid)
}

pub(super) fn logical(
    values: Vec<Value>,
    shape: Vec<usize>,
    is_sparse: bool,
) -> BuiltinResult<Value> {
    require_dense(is_sparse)?;
    let flags = values
        .into_iter()
        .map(|value| match value {
            Value::Bool(flag) => Ok(u8::from(flag)),
            _ => Err(class_mismatch()),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    LogicalArray::new(flags, shape)
        .map(Value::LogicalArray)
        .map_err(error::invalid)
}

pub(super) fn floating(
    values: Vec<Value>,
    shape: Vec<usize>,
    is_sparse: bool,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    if is_sparse && dtype != NumericDType::F64 {
        return Err(error::invalid(
            "accumarray: sparse output requires double scalar group results",
        ));
    }
    let values = values
        .iter()
        .map(|value| data::as_numeric_scalar(value).ok_or_else(class_mismatch))
        .collect::<BuiltinResult<Vec<_>>>()?;
    if is_sparse {
        sparse::numeric(values, shape, true)
    } else {
        Tensor::new_with_dtype(values, shape, dtype)
            .map(Value::Tensor)
            .map_err(error::invalid)
    }
}

pub(super) fn exact_integers(values: &[Value]) -> Option<Vec<IntValue>> {
    values.iter().map(exact_integer).collect()
}

pub(super) fn exact_integer(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            tensor.integer_storage()?.value_at(0)
        }
        _ => None,
    }
}

pub(super) fn common_floating_dtype(values: &[Value]) -> Option<NumericDType> {
    let dtypes = values
        .iter()
        .map(floating_dtype)
        .collect::<Option<Vec<_>>>()?;
    let first = dtypes.first().copied()?;
    dtypes.iter().all(|dtype| *dtype == first).then_some(first)
}

pub(super) fn floating_dtype(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Num(_) => Some(NumericDType::F64),
        Value::Tensor(tensor)
            if tensor_utils::is_scalar_tensor(tensor) && tensor.integer_storage().is_none() =>
        {
            Some(tensor.numeric_dtype())
        }
        _ => None,
    }
}

pub(super) fn class_mismatch() -> crate::RuntimeError {
    error::invalid("accumarray: fill value class must match group output")
}

fn require_dense(sparse: bool) -> BuiltinResult<()> {
    if sparse {
        Err(error::invalid(
            "accumarray: sparse output requires double scalar group results",
        ))
    } else {
        Ok(())
    }
}
