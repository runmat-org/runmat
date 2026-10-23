use super::single_error_with_detail;
use crate::BuiltinResult;
use runmat_builtins::SINGLE_ERROR_INTERNAL;
use runmat_value::{CharArray, ComplexTensor, IntValue, NumericScalar, NumericStorage, Tensor};

pub(super) fn single_tensor_to_prepared_host(
    runtime: &crate::context::RuntimeContext,
    tensor: Tensor,
    mut output: crate::context::preparation::PreparedHostNumericalOutput,
) -> BuiltinResult<Tensor> {
    if !output.matches_floating_cast(
        runmat_types::NumericClass::Single,
        runmat_value::NumericDType::F32,
        tensor.shape(),
    ) {
        return Err(single_error_with_detail(
            &SINGLE_ERROR_INTERNAL,
            "selected exact single output does not match the prepared source",
        ));
    }
    let copied = match (tensor.as_f32_slice(), tensor.as_f64_slice()) {
        (Some(source), _) => output.copy_f32_from_f32(runtime, source),
        (_, Some(source)) => output.copy_f32_from_f64(runtime, source),
        _ => {
            return Err(single_error_with_detail(
                &SINGLE_ERROR_INTERNAL,
                "selected exact single source has an unsupported native class",
            ));
        }
    };
    copied.map_err(|error| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            SINGLE_ERROR_INTERNAL
                .identifier
                .unwrap_or("RunMat:single:Internal"),
            "selected exact single output authority changed",
            error,
        )
    })?;
    output.into_tensor(runtime).map_err(|error| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            SINGLE_ERROR_INTERNAL
                .identifier
                .unwrap_or("RunMat:single:Internal"),
            "selected exact single output could not be published",
            error,
        )
    })
}

pub(super) fn single_tensor_to_host(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| single_error_with_detail(&SINGLE_ERROR_INTERNAL, error))?;
    let values = storage.materialize_f32();
    Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
        .map_err(|error| single_error_with_detail(&SINGLE_ERROR_INTERNAL, error))
}

pub(super) fn single_complex_tensor_to_host(tensor: ComplexTensor) -> BuiltinResult<ComplexTensor> {
    let shape = tensor.shape().to_vec();
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

pub(crate) fn numeric_scalar_to_f32(value: NumericScalar) -> f32 {
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
    let ascii: Vec<f64> = chars
        .to_column_major()
        .into_iter()
        .map(|ch| f64::from(ch.raw()))
        .collect();
    Tensor::new(ascii, chars.shape.clone())
        .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))
}
