//! Numeric-class selection and storage for digit conversion results.

use super::bi2de_error;
use crate::BuiltinResult;
use runmat_value::{IntValue, NumericDType, NumericStorage, Tensor, Value};

pub(super) fn digit_decimal_result(
    values: Vec<f64>,
    input_dtype: Option<NumericDType>,
) -> BuiltinResult<Value> {
    let dtype = match input_dtype {
        Some(NumericDType::I8 | NumericDType::I16 | NumericDType::I32 | NumericDType::I64) => {
            let maximum = values.iter().copied().fold(0.0, f64::max);
            let minimum = input_dtype.expect("matched integer input");
            if minimum == NumericDType::I8 && maximum <= i8::MAX as f64 {
                NumericDType::I8
            } else if matches!(minimum, NumericDType::I8 | NumericDType::I16)
                && maximum <= i16::MAX as f64
            {
                NumericDType::I16
            } else if minimum != NumericDType::I64 && maximum <= i32::MAX as f64 {
                NumericDType::I32
            } else {
                NumericDType::I64
            }
        }
        Some(NumericDType::U8 | NumericDType::U16 | NumericDType::U32 | NumericDType::U64) => {
            let maximum = values.iter().copied().fold(0.0, f64::max);
            let minimum = input_dtype.expect("matched integer input");
            if minimum == NumericDType::U8 && maximum <= u8::MAX as f64 {
                NumericDType::U8
            } else if matches!(minimum, NumericDType::U8 | NumericDType::U16)
                && maximum <= u16::MAX as f64
            {
                NumericDType::U16
            } else if minimum != NumericDType::U64 && maximum <= u32::MAX as f64 {
                NumericDType::U32
            } else {
                NumericDType::U64
            }
        }
        Some(dtype) => dtype,
        None => NumericDType::F64,
    };
    if values.len() == 1 {
        let value = values[0];
        return Ok(match dtype {
            NumericDType::F64 => Value::Num(value),
            NumericDType::F32 => Value::Tensor(
                Tensor::from_f32(vec![value as f32], vec![1, 1])
                    .map_err(|error| bi2de_error(format!("bi2de: {error}")))?,
            ),
            NumericDType::I8 => Value::Int(IntValue::I8(value as i8)),
            NumericDType::I16 => Value::Int(IntValue::I16(value as i16)),
            NumericDType::I32 => Value::Int(IntValue::I32(value as i32)),
            NumericDType::I64 => Value::Int(IntValue::I64(value as i64)),
            NumericDType::U8 => Value::Int(IntValue::U8(value as u8)),
            NumericDType::U16 => Value::Int(IntValue::U16(value as u16)),
            NumericDType::U32 => Value::Int(IntValue::U32(value as u32)),
            NumericDType::U64 => Value::Int(IntValue::U64(value as u64)),
        });
    }
    let rows = values.len();
    let storage = digit_output_storage(dtype, values);
    Ok(Value::Tensor(
        Tensor::from_numeric_storage(storage, vec![rows, 1])
            .map_err(|error| bi2de_error(format!("bi2de: {error}")))?,
    ))
}

pub(super) fn digit_output_storage(dtype: NumericDType, values: Vec<f64>) -> NumericStorage {
    macro_rules! storage {
        ($variant:ident, $type:ty) => {
            NumericStorage::$variant(values.into_iter().map(|value| value as $type).collect())
        };
    }
    match dtype {
        NumericDType::F64 => NumericStorage::F64(values),
        NumericDType::F32 => storage!(F32, f32),
        NumericDType::I8 => storage!(I8, i8),
        NumericDType::I16 => storage!(I16, i16),
        NumericDType::I32 => storage!(I32, i32),
        NumericDType::I64 => storage!(I64, i64),
        NumericDType::U8 => storage!(U8, u8),
        NumericDType::U16 => storage!(U16, u16),
        NumericDType::U32 => storage!(U32, u32),
        NumericDType::U64 => storage!(U64, u64),
    }
}
