use super::single_error_with_detail;
use crate::BuiltinResult;
use runmat_builtins::SINGLE_ERROR_INTERNAL;
use runmat_value::{CharArray, ComplexTensor, IntValue, NumericScalar, NumericStorage, Tensor};

pub(super) fn single_tensor_to_host(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| single_error_with_detail(&SINGLE_ERROR_INTERNAL, error))?;
    let values = storage.materialize_f32();
    Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
        .map_err(|error| single_error_with_detail(&SINGLE_ERROR_INTERNAL, error))
}

pub(super) fn single_complex_tensor_to_host(tensor: ComplexTensor) -> BuiltinResult<ComplexTensor> {
    let shape = tensor.shape.clone();
    let data = (0..tensor.len())
        .map(|index| {
            tensor
                .numeric_value_at(index)
                .map(|(real, imag)| (numeric_scalar_to_f32(real), numeric_scalar_to_f32(imag)))
                .ok_or_else(|| {
                    single_error_with_detail(
                        &SINGLE_ERROR_INTERNAL,
                        "complex value storage is inconsistent",
                    )
                })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    ComplexTensor::from_f32(data, shape)
        .map_err(|error| single_error_with_detail(&SINGLE_ERROR_INTERNAL, error))
}

pub(super) fn int_value_to_f32(value: &IntValue) -> f32 {
    match value {
        IntValue::I8(value) => f32::from(*value),
        IntValue::I16(value) => f32::from(*value),
        IntValue::I32(value) => *value as f32,
        IntValue::I64(value) => *value as f32,
        IntValue::U8(value) => f32::from(*value),
        IntValue::U16(value) => f32::from(*value),
        IntValue::U32(value) => *value as f32,
        IntValue::U64(value) => *value as f32,
    }
}

pub(super) fn numeric_scalar_to_f32(value: NumericScalar) -> f32 {
    match value {
        NumericScalar::F64(value) => value as f32,
        NumericScalar::F32(value) => value,
        NumericScalar::I8(value) => f32::from(value),
        NumericScalar::I16(value) => f32::from(value),
        NumericScalar::I32(value) => value as f32,
        NumericScalar::I64(value) => value as f32,
        NumericScalar::U8(value) => f32::from(value),
        NumericScalar::U16(value) => f32::from(value),
        NumericScalar::U32(value) => value as f32,
        NumericScalar::U64(value) => value as f32,
    }
}

pub(super) fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let ascii: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(ascii, vec![chars.rows, chars.cols])
        .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))
}
