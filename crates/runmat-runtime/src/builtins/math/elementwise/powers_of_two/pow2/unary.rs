use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::{random_args::complex_tensor_into_value, tensor};
use crate::BuiltinResult;

use super::{errors, input, numeric, provider, BUILTIN_NAME};

pub(super) async fn evaluate(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => provider::evaluate_unary(handle).await,
        other => evaluate_host(other),
    }
}

pub(super) fn evaluate_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(real, imaginary) => {
            let (real, imaginary) = numeric::power_f64(real, imaginary);
            Ok(Value::Complex(real, imaginary))
        }
        Value::ComplexTensor(tensor) => transform_complex(tensor),
        Value::CharArray(chars) => transform_characters(chars),
        Value::String(_) | Value::StringArray(_) => {
            Err(errors::invalid_input("expected numeric input, got string"))
        }
        Value::GpuTensor(_) => Err(errors::internal(
            "resident input reached the host unary pow2 path",
        )),
        other => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, other)
                .map_err(errors::invalid_input)?;
            Ok(tensor::tensor_into_value(transform_real(tensor)?))
        }
    }
}

pub(super) fn transform_real(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(errors::internal)?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(f64::exp2).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(f32::exp2).collect())
        }
        storage => NumericStorage::F64(
            input::integer_values_as_f64(storage)?
                .into_iter()
                .map(f64::exp2)
                .collect(),
        ),
    };
    Tensor::from_numeric_storage(output, shape).map_err(errors::internal)
}

fn transform_complex(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = match tensor.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imaginary)| numeric::power_f64(real, imaginary))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imaginary)| numeric::power_f32(real, imaginary))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(errors::invalid_input(
                "complex fixed-width integer input is not supported",
            ))
        }
    };
    ComplexTensor::from_complex_storage(storage, shape)
        .map(complex_tensor_into_value)
        .map_err(errors::internal)
}

fn transform_characters(chars: CharArray) -> BuiltinResult<Value> {
    let shape = vec![chars.rows, chars.cols];
    let values = chars
        .data
        .into_iter()
        .map(|character| (character as u32 as f64).exp2())
        .collect();
    Tensor::new(values, shape)
        .map(Value::Tensor)
        .map_err(errors::internal)
}
