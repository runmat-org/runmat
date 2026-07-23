//! Shared exact host-side conversion support for MATLAB integer cast builtins.

use runmat_builtins::{IntValue, IntegerStorage, Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntegerTarget {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntegerTarget {
    pub(crate) fn from_int_value(value: &IntValue) -> Self {
        match value {
            IntValue::I8(_) => Self::I8,
            IntValue::I16(_) => Self::I16,
            IntValue::I32(_) => Self::I32,
            IntValue::I64(_) => Self::I64,
            IntValue::U8(_) => Self::U8,
            IntValue::U16(_) => Self::U16,
            IntValue::U32(_) => Self::U32,
            IntValue::U64(_) => Self::U64,
        }
    }

    pub(crate) fn from_storage(storage: &IntegerStorage) -> Self {
        match storage {
            IntegerStorage::I8(_) => Self::I8,
            IntegerStorage::I16(_) => Self::I16,
            IntegerStorage::I32(_) => Self::I32,
            IntegerStorage::I64(_) => Self::I64,
            IntegerStorage::U8(_) => Self::U8,
            IntegerStorage::U16(_) => Self::U16,
            IntegerStorage::U32(_) => Self::U32,
            IntegerStorage::U64(_) => Self::U64,
        }
    }

    pub(crate) fn cast_scalar(self, value: f64) -> IntValue {
        match self {
            Self::I8 => IntValue::I8(cast_signed(value, i8::MIN as f64, i8::MAX as f64) as i8),
            Self::I16 => IntValue::I16(cast_signed(value, i16::MIN as f64, i16::MAX as f64) as i16),
            Self::I32 => IntValue::I32(cast_signed(value, i32::MIN as f64, i32::MAX as f64) as i32),
            Self::I64 => IntValue::I64(cast_signed(value, i64::MIN as f64, i64::MAX as f64)),
            Self::U8 => IntValue::U8(cast_unsigned(value, u8::MAX as f64) as u8),
            Self::U16 => IntValue::U16(cast_unsigned(value, u16::MAX as f64) as u16),
            Self::U32 => IntValue::U32(cast_unsigned(value, u32::MAX as f64) as u32),
            Self::U64 => IntValue::U64(cast_unsigned(value, u64::MAX as f64)),
        }
    }

    pub(crate) fn cast_int(self, value: &IntValue) -> IntValue {
        match self {
            Self::I8 => IntValue::I8(value.to_i64().clamp(i8::MIN as i64, i8::MAX as i64) as i8),
            Self::I16 => {
                IntValue::I16(value.to_i64().clamp(i16::MIN as i64, i16::MAX as i64) as i16)
            }
            Self::I32 => {
                IntValue::I32(value.to_i64().clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            }
            Self::I64 => IntValue::I64(value.to_i64()),
            Self::U8 => IntValue::U8(unsigned_value(value).min(u8::MAX as u64) as u8),
            Self::U16 => IntValue::U16(unsigned_value(value).min(u16::MAX as u64) as u16),
            Self::U32 => IntValue::U32(unsigned_value(value).min(u32::MAX as u64) as u32),
            Self::U64 => IntValue::U64(unsigned_value(value)),
        }
    }

    pub(crate) fn cast_tensor(self, tensor: Tensor) -> Result<Tensor, String> {
        let values = match tensor.integer_data {
            Some(storage) => integer_values(storage)
                .iter()
                .map(|value| self.cast_int(value))
                .collect(),
            None => tensor
                .data
                .iter()
                .map(|&value| self.cast_scalar(value))
                .collect(),
        };
        Tensor::new_integer(self.storage(values), tensor.shape)
    }

    pub(crate) fn storage(self, values: Vec<IntValue>) -> IntegerStorage {
        match self {
            Self::I8 => IntegerStorage::I8(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::I8(value) => value,
                        _ => unreachable!("target conversion must produce int8"),
                    })
                    .collect(),
            ),
            Self::I16 => IntegerStorage::I16(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::I16(value) => value,
                        _ => unreachable!("target conversion must produce int16"),
                    })
                    .collect(),
            ),
            Self::I32 => IntegerStorage::I32(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::I32(value) => value,
                        _ => unreachable!("target conversion must produce int32"),
                    })
                    .collect(),
            ),
            Self::I64 => IntegerStorage::I64(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::I64(value) => value,
                        _ => unreachable!("target conversion must produce int64"),
                    })
                    .collect(),
            ),
            Self::U8 => IntegerStorage::U8(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::U8(value) => value,
                        _ => unreachable!("target conversion must produce uint8"),
                    })
                    .collect(),
            ),
            Self::U16 => IntegerStorage::U16(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::U16(value) => value,
                        _ => unreachable!("target conversion must produce uint16"),
                    })
                    .collect(),
            ),
            Self::U32 => IntegerStorage::U32(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::U32(value) => value,
                        _ => unreachable!("target conversion must produce uint32"),
                    })
                    .collect(),
            ),
            Self::U64 => IntegerStorage::U64(
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::U64(value) => value,
                        _ => unreachable!("target conversion must produce uint64"),
                    })
                    .collect(),
            ),
        }
    }
}

pub(crate) enum CastError {
    Unsupported(String),
    Internal(String),
}

pub(crate) async fn cast_value(value: Value, target: IntegerTarget) -> Result<Value, CastError> {
    match value {
        Value::Num(value) => Ok(Value::Int(target.cast_scalar(value))),
        Value::Int(value) => Ok(Value::Int(target.cast_int(&value))),
        Value::Bool(value) => Ok(Value::Int(target.cast_scalar(if value {
            1.0
        } else {
            0.0
        }))),
        Value::Tensor(tensor) => cast_tensor_value(target, tensor),
        Value::SparseTensor(_) => Err(CastError::Unsupported("sparse".to_string())),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array).map_err(CastError::Internal)?;
            cast_tensor_value(target, tensor)
        }
        Value::CharArray(chars) => {
            let tensor = Tensor::new(
                chars
                    .data
                    .iter()
                    .map(|&value| value as u32 as f64)
                    .collect(),
                vec![chars.rows, chars.cols],
            )
            .map_err(CastError::Internal)?;
            cast_tensor_value(target, tensor)
        }
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|error| CastError::Internal(error.message))?;
            cast_tensor_value(target, tensor)
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(CastError::Unsupported("complex".to_string()))
        }
        Value::String(_) | Value::StringArray(_) => {
            Err(CastError::Unsupported("string".to_string()))
        }
        Value::Symbolic(expression) => expression
            .numeric_constant_value()
            .map(|value| Value::Int(target.cast_scalar(value)))
            .ok_or_else(|| CastError::Unsupported("sym".to_string())),
        Value::Cell(_) => Err(CastError::Unsupported("cell".to_string())),
        Value::Struct(_) => Err(CastError::Unsupported("struct".to_string())),
        Value::Object(object) => Err(CastError::Unsupported(object.class_name)),
        Value::HandleObject(handle) => Err(CastError::Unsupported(handle.class_name)),
        Value::Listener(_) => Err(CastError::Unsupported("event.listener".to_string())),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(CastError::Unsupported("function_handle".to_string())),
        Value::ClassRef(_) => Err(CastError::Unsupported("meta.class".to_string())),
        Value::MException(_) => Err(CastError::Unsupported("MException".to_string())),
        Value::OutputList(_) => Err(CastError::Unsupported("OutputList".to_string())),
    }
}

fn cast_tensor_value(target: IntegerTarget, tensor: Tensor) -> Result<Value, CastError> {
    let tensor = target.cast_tensor(tensor).map_err(CastError::Internal)?;
    if tensor.data.len() == 1 {
        let storage = tensor
            .integer_data
            .expect("integer cast must construct exact integer storage");
        Ok(Value::Int(
            integer_values(storage).pop().expect("scalar storage"),
        ))
    } else {
        Ok(Value::Tensor(tensor))
    }
}

fn cast_signed(value: f64, min: f64, max: f64) -> i64 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            min as i64
        } else {
            max as i64
        }
    } else {
        value.round().clamp(min, max) as i64
    }
}

fn cast_unsigned(value: f64, max: f64) -> u64 {
    if value.is_nan() || value.is_sign_negative() {
        0
    } else if value.is_infinite() {
        max as u64
    } else {
        value.round().clamp(0.0, max) as u64
    }
}

fn unsigned_value(value: &IntValue) -> u64 {
    match value {
        IntValue::U64(value) => *value,
        _ => value.to_i64().max(0) as u64,
    }
}

pub(crate) fn integer_values(storage: IntegerStorage) -> Vec<IntValue> {
    match storage {
        IntegerStorage::I8(values) => values.into_iter().map(IntValue::I8).collect(),
        IntegerStorage::I16(values) => values.into_iter().map(IntValue::I16).collect(),
        IntegerStorage::I32(values) => values.into_iter().map(IntValue::I32).collect(),
        IntegerStorage::I64(values) => values.into_iter().map(IntValue::I64).collect(),
        IntegerStorage::U8(values) => values.into_iter().map(IntValue::U8).collect(),
        IntegerStorage::U16(values) => values.into_iter().map(IntValue::U16).collect(),
        IntegerStorage::U32(values) => values.into_iter().map(IntValue::U32).collect(),
        IntegerStorage::U64(values) => values.into_iter().map(IntValue::U64).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uint64_to_int64_array_saturates_without_f64_rounding() {
        let source = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("source tensor");
        let output = IntegerTarget::I64
            .cast_tensor(source)
            .expect("int64 conversion");

        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MAX]))
        );
    }

    #[test]
    fn int64_to_uint64_array_clamps_negative_values_exactly() {
        let source = Tensor::new_integer(IntegerStorage::I64(vec![-1, i64::MAX]), vec![1, 2])
            .expect("source tensor");
        let output = IntegerTarget::U64
            .cast_tensor(source)
            .expect("uint64 conversion");

        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![0, i64::MAX as u64]))
        );
    }
}
