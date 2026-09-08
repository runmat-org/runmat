use runmat_value::{
    ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray, NumericDType, Tensor, Value,
};

use super::super::error::{mat2cell_error_with_message, MAT2CELL_ERROR_INTERNAL};
use super::super::extract;
use crate::builtins::common::tensor;
use crate::BuiltinResult;

pub(super) fn real(
    tensor: &Tensor,
    shape: &[usize],
    start: &[usize],
    sizes: &[usize],
) -> BuiltinResult<Value> {
    if let Some(storage) = tensor.integer_storage() {
        return integer(storage, shape, start, sizes);
    }
    let output_shape = extract::output_shape(sizes);
    let tensor = match tensor.numeric_dtype() {
        NumericDType::F64 => Tensor::new(
            extract::block(
                tensor.as_f64_slice().expect("double storage"),
                shape,
                start,
                sizes,
            )?,
            output_shape,
        ),
        NumericDType::F32 => {
            let values = (0..tensor.len())
                .map(|index| match tensor.numeric_value_at(index) {
                    Some(runmat_value::NumericScalar::F32(value)) => value,
                    _ => unreachable!("single storage"),
                })
                .collect::<Vec<_>>();
            Tensor::from_f32(extract::block(&values, shape, start, sizes)?, output_shape)
        }
        _ => unreachable!("integer storage handled before floating dispatch"),
    }
    .map_err(internal)?;
    Ok(tensor::tensor_into_value(tensor))
}

pub(super) fn complex(
    tensor: &ComplexTensor,
    shape: &[usize],
    start: &[usize],
    sizes: &[usize],
) -> BuiltinResult<Value> {
    if let Some(storage) = tensor.integer_storage() {
        return complex_integer(storage, shape, start, sizes);
    }
    let data = extract::block(&tensor.materialize_f64(), shape, start, sizes)?;
    if data.len() == 1 {
        return Ok(Value::Complex(data[0].0, data[0].1));
    }
    ComplexTensor::new(data, extract::output_shape(sizes))
        .map(Value::ComplexTensor)
        .map_err(internal)
}

pub(super) fn logical(
    array: &LogicalArray,
    shape: &[usize],
    start: &[usize],
    sizes: &[usize],
) -> BuiltinResult<Value> {
    let data = extract::block(&array.data, shape, start, sizes)?;
    if data.len() == 1 {
        return Ok(Value::Bool(data[0] != 0));
    }
    LogicalArray::new(data, extract::output_shape(sizes))
        .map(Value::LogicalArray)
        .map_err(internal)
}

fn integer(
    storage: &IntegerStorage,
    shape: &[usize],
    start: &[usize],
    sizes: &[usize],
) -> BuiltinResult<Value> {
    let values = extract::block(&storage.exact_values(), shape, start, sizes)?;
    let storage = storage.from_exact_values_like(values).map_err(internal)?;
    let tensor = Tensor::new_integer(storage, extract::output_shape(sizes)).map_err(internal)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn complex_integer(
    storage: &IntegerComplexStorage,
    shape: &[usize],
    start: &[usize],
    sizes: &[usize],
) -> BuiltinResult<Value> {
    let real = storage
        .real
        .from_exact_values_like(extract::block(
            &storage.real.exact_values(),
            shape,
            start,
            sizes,
        )?)
        .map_err(internal)?;
    let imaginary = storage
        .imag
        .from_exact_values_like(extract::block(
            &storage.imag.exact_values(),
            shape,
            start,
            sizes,
        )?)
        .map_err(internal)?;
    let storage = IntegerComplexStorage::new(real, imaginary).map_err(internal)?;
    ComplexTensor::new_integer(storage, extract::output_shape(sizes))
        .map(Value::ComplexTensor)
        .map_err(internal)
}

fn internal(error: impl std::fmt::Display) -> crate::RuntimeError {
    mat2cell_error_with_message(format!("mat2cell: {error}"), &MAT2CELL_ERROR_INTERNAL)
}
