use runmat_value::{CharArray, NumericStorage, Tensor, Value};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::{errors, BUILTIN_NAME};

pub(super) fn execute(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::CharArray(array) => character(array),
        value => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
                .map_err(errors::invalid_input)?;
            Ok(tensor::tensor_into_value(apply(tensor)?))
        }
    }
}

pub(super) fn apply(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(errors::internal)?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(scalar).collect())
        }
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| scalar(value.into()) as f32)
                .collect(),
        ),
        NumericStorage::I8(values) => NumericStorage::F64(integer(values)),
        NumericStorage::I16(values) => NumericStorage::F64(integer(values)),
        NumericStorage::I32(values) => NumericStorage::F64(integer(values)),
        NumericStorage::I64(values) => NumericStorage::F64(integer(values)),
        NumericStorage::U8(values) => NumericStorage::F64(integer(values)),
        NumericStorage::U16(values) => NumericStorage::F64(integer(values)),
        NumericStorage::U32(values) => NumericStorage::F64(integer(values)),
        NumericStorage::U64(values) => NumericStorage::F64(integer(values)),
    };
    Tensor::from_numeric_storage(output, shape).map_err(errors::internal)
}

fn integer<T>(values: Vec<T>) -> Vec<f64>
where
    T: PartialOrd + Default,
{
    let zero = T::default();
    values
        .into_iter()
        .map(|value| {
            if value > zero {
                1.0
            } else if value < zero {
                0.0
            } else {
                0.5
            }
        })
        .collect()
}

fn character(array: CharArray) -> BuiltinResult<Value> {
    let values = array
        .data
        .iter()
        .map(|character| scalar(f64::from(u32::from(*character))))
        .collect();
    Tensor::new(values, vec![array.rows, array.cols])
        .map(Value::Tensor)
        .map_err(errors::internal)
}

#[inline]
pub(super) fn scalar(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        0.0
    } else if value == 0.0 {
        0.5
    } else {
        value
    }
}
