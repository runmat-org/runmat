use runmat_builtins::{HYPOT_ERROR_INTERNAL, HYPOT_ERROR_INVALID_INPUT};
use runmat_value::{ComplexStorage, NumericStorage, Tensor, Value};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::super::errors;
use super::kernel;

pub(super) fn into_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::CharArray(characters) => {
            let data = characters
                .data
                .iter()
                .map(|&character| character as u32 as f64)
                .collect();
            Tensor::new(data, vec![characters.rows, characters.cols])
                .map_err(|error| errors::with_detail(&HYPOT_ERROR_INTERNAL, error))
        }
        Value::Complex(real, imaginary) => {
            Tensor::new(vec![kernel::complex_magnitude(real, imaginary)], vec![1, 1])
                .map_err(|error| errors::with_detail(&HYPOT_ERROR_INTERNAL, error))
        }
        Value::ComplexTensor(tensor) => complex_tensor_into_magnitudes(tensor),
        Value::GpuTensor(_) => Err(errors::with_detail(
            &HYPOT_ERROR_INTERNAL,
            "internal error converting GPU tensor",
        )),
        other => tensor::value_into_tensor_for("hypot", other)
            .map_err(|error| errors::with_detail(&HYPOT_ERROR_INVALID_INPUT, error)),
    }
}

pub(super) fn single_domain(storage: NumericStorage) -> Vec<f32> {
    storage.materialize_f32()
}

pub(super) fn double_domain(storage: NumericStorage) -> BuiltinResult<Vec<f64>> {
    match storage {
        NumericStorage::F64(values) => Ok(values),
        NumericStorage::F32(values) => Ok(values.into_iter().map(f64::from).collect()),
        storage => storage
            .into_integer_storage()
            .map(|integer| integer.to_f64_vec())
            .map_err(|_| {
                errors::with_detail(
                    &HYPOT_ERROR_INTERNAL,
                    "unsupported numeric storage at the binary64 calculation boundary",
                )
            }),
    }
}

pub(in super::super) fn scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(number) => Some(*number),
        Value::Int(integer) => Some(integer.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Some(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        Value::CharArray(characters) if characters.rows * characters.cols == 1 => characters
            .data
            .first()
            .map(|&character| character as u32 as f64)
            .or(Some(0.0)),
        Value::Complex(real, imaginary) => Some(kernel::complex_magnitude(*real, *imaginary)),
        _ => None,
    }
}

fn complex_tensor_into_magnitudes(tensor: runmat_value::ComplexTensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = match tensor.into_complex_storage() {
        ComplexStorage::F64(values) => NumericStorage::F64(
            values
                .into_iter()
                .map(|(real, imaginary)| kernel::f64(real, imaginary))
                .collect(),
        ),
        ComplexStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|(real, imaginary)| kernel::f32(real, imaginary))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(errors::with_detail(
                &HYPOT_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ));
        }
    };
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|error| errors::with_detail(&HYPOT_ERROR_INTERNAL, error))
}
