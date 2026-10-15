use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::{build_runtime_error, BuiltinResult};

pub(super) mod resident_range;
pub(super) mod fusion;
pub(crate) mod preparation;
pub(super) mod like;
pub(super) mod resident_complex;
pub(super) mod real_fallback;

#[cfg(test)]
mod domain_preparation_tests;

pub(crate) fn ensure_input_extensions(
    value: &Value,
    builtin: &str,
    integer: &'static BuiltinExtensionDescriptor,
    logical: &'static BuiltinExtensionDescriptor,
    character: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    let is_integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some()
        );
    if is_integer {
        crate::compatibility::ensure_builtin_extension_enabled(integer, builtin)?;
    }
    let is_logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(
            value,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle)
        );
    if is_logical {
        crate::compatibility::ensure_builtin_extension_enabled(logical, builtin)?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(character, builtin)?;
    }
    Ok(())
}

pub(crate) fn reject_excess_outputs(builtin: &str) -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(
            build_runtime_error(format!("{builtin}: too many output arguments"))
                .with_builtin(builtin)
                .with_identifier(format!("RunMat:{builtin}:TooManyOutputs"))
                .build(),
        );
    }
    Ok(())
}

pub(crate) fn ensure_integer_exact_f64(value: &Value, builtin: &str) -> BuiltinResult<()> {
    let exact = crate::builtins::common::validation::integer_is_exact_f64;
    let valid = match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(build_runtime_error(format!(
            "{builtin}: integer input must be exactly representable as double"
        ))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:InvalidInput"))
        .build())
    }
}

pub(crate) fn map_real_tensor<F64, F32>(
    tensor: Tensor,
    builtin: &str,
    map_f64: F64,
    map_f32: F32,
) -> BuiltinResult<Tensor>
where
    F64: Fn(f64) -> f64,
    F32: Fn(f32) -> f32,
{
    let shape = tensor.shape().to_vec();
    let storage = tensor.into_numeric_storage().map_err(|error| {
        build_runtime_error(format!("{builtin}: {error}"))
            .with_builtin(builtin)
            .build()
    })?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(map_f64).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(map_f32).collect())
        }
        integer => NumericStorage::F64(
            integer
                .into_integer_storage()
                .expect("inverse function integer boundary")
                .to_f64_vec()
                .into_iter()
                .map(map_f64)
                .collect(),
        ),
    };
    Tensor::from_numeric_storage(output, shape).map_err(|error| {
        build_runtime_error(format!("{builtin}: {error}"))
            .with_builtin(builtin)
            .build()
    })
}

pub(crate) fn map_complex_tensor<F64, F32>(
    tensor: ComplexTensor,
    builtin: &str,
    map_f64: F64,
    map_f32: F32,
) -> BuiltinResult<ComplexTensor>
where
    F64: Fn((f64, f64)) -> (f64, f64),
    F32: Fn((f32, f32)) -> (f32, f32),
{
    let (storage, shape) = tensor.into_complex_storage_and_shape();
    let storage = match storage {
        ComplexStorage::F64(values) => {
            ComplexStorage::F64(values.into_iter().map(map_f64).collect())
        }
        ComplexStorage::F32(values) => {
            ComplexStorage::F32(values.into_iter().map(map_f32).collect())
        }
        ComplexStorage::Integer(_) => {
            return Err(build_runtime_error(format!(
                "{builtin}: typed complex integer input is not supported"
            ))
            .with_builtin(builtin)
            .build())
        }
    };
    ComplexTensor::from_complex_storage(storage, shape).map_err(|error| {
        build_runtime_error(format!("{builtin}: {error}"))
            .with_builtin(builtin)
            .build()
    })
}

pub(crate) fn map_real_tensor_promoting<F64, F32>(
    tensor: Tensor,
    builtin: &str,
    map_f64: F64,
    map_f32: F32,
) -> BuiltinResult<Value>
where
    F64: Fn(f64) -> (f64, f64),
    F32: Fn(f32) -> (f32, f32),
{
    let shape = tensor.shape().to_vec();
    let storage = tensor.into_numeric_storage().map_err(|error| {
        build_runtime_error(format!("{builtin}: {error}"))
            .with_builtin(builtin)
            .build()
    })?;
    match storage {
        NumericStorage::F64(values) => map_promoting_f64(values, shape, builtin, &map_f64),
        NumericStorage::F32(values) => map_promoting_f32(values, shape, builtin, &map_f32),
        integer => map_promoting_f64(
            integer
                .into_integer_storage()
                .expect("inverse function integer boundary")
                .to_f64_vec(),
            shape,
            builtin,
            &map_f64,
        ),
    }
}

fn map_promoting_f64<F>(
    values: Vec<f64>,
    shape: Vec<usize>,
    builtin: &str,
    map: &F,
) -> BuiltinResult<Value>
where
    F: Fn(f64) -> (f64, f64),
{
    if values.is_empty() {
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F64(values), shape).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        return Ok(crate::builtins::common::tensor::tensor_into_value(tensor));
    }
    let mapped = values.into_iter().map(map).collect::<Vec<_>>();
    if mapped.iter().any(|(_, imag)| *imag != 0.0) {
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F64(mapped.into()), shape)
            .map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = mapped.into_iter().map(|(real, _)| real).collect();
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F64(values), shape).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        Ok(crate::builtins::common::tensor::tensor_into_value(tensor))
    }
}

fn map_promoting_f32<F>(
    values: Vec<f32>,
    shape: Vec<usize>,
    builtin: &str,
    map: &F,
) -> BuiltinResult<Value>
where
    F: Fn(f32) -> (f32, f32),
{
    if values.is_empty() {
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F32(values), shape).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        return Ok(crate::builtins::common::tensor::tensor_into_value(tensor));
    }
    let mapped = values.into_iter().map(map).collect::<Vec<_>>();
    if mapped.iter().any(|(_, imag)| *imag != 0.0) {
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F32(mapped.into()), shape)
            .map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = mapped.into_iter().map(|(real, _)| real).collect();
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F32(values), shape).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        Ok(crate::builtins::common::tensor::tensor_into_value(tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntegerStorage, NumericDType};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn unary_mapping_keeps_low_rank_shape_and_native_precision() {
        for (shape, values) in [(Vec::new(), vec![1.0]), (vec![2], vec![1.0, 2.0])] {
            for single in [false, true] {
                let real = if single {
                    Tensor::from_f32(
                        values.iter().map(|value| *value as f32).collect(),
                        shape.clone(),
                    )
                    .unwrap()
                } else {
                    Tensor::new(values.clone(), shape.clone()).unwrap()
                };
                let dtype = real.numeric_dtype();
                let output =
                    map_real_tensor(real, "mapping-test", |value| value, |value| value).unwrap();
                assert_eq!(output.shape(), shape.as_slice());
                assert_eq!(output.numeric_dtype(), dtype);
                assert_eq!(output.materialize_f64(), values);

                let complex_values = values.iter().map(|value| (*value, 1.0)).collect::<Vec<_>>();
                let complex = if single {
                    ComplexTensor::from_f32(
                        complex_values
                            .iter()
                            .map(|(real, imaginary)| (*real as f32, *imaginary as f32))
                            .collect(),
                        shape.clone(),
                    )
                    .unwrap()
                } else {
                    ComplexTensor::new(complex_values.clone(), shape.clone()).unwrap()
                };
                let output =
                    map_complex_tensor(complex, "mapping-test", |value| value, |value| value)
                        .unwrap();
                assert_eq!(output.shape(), shape.as_slice());
                assert_eq!(output.numeric_dtype(), dtype);
                assert_eq!(output.materialize_f64(), complex_values);
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn promoting_mapping_preserves_rank_one_real_and_complex_classes() {
        for imaginary in [0.0, 1.0] {
            let input = Tensor::from_f32(vec![1.0, 2.0], vec![2]).unwrap();
            let output = map_real_tensor_promoting(
                input,
                "mapping-test",
                |value| (value, imaginary),
                |value| (value, imaginary as f32),
            )
            .unwrap();
            match output {
                Value::Tensor(output) => {
                    assert_eq!(imaginary, 0.0);
                    assert_eq!(output.shape(), &[2]);
                    assert_eq!(output.numeric_dtype(), NumericDType::F32);
                    assert_eq!(output.as_f32_slice(), Some([1.0, 2.0].as_slice()));
                }
                Value::ComplexTensor(output) => {
                    assert_eq!(imaginary, 1.0);
                    assert_eq!(output.shape(), &[2]);
                    assert_eq!(output.numeric_dtype(), NumericDType::F32);
                    assert_eq!(output.materialize_f64(), vec![(1.0, 1.0), (2.0, 1.0)]);
                }
                other => panic!("expected mapped array, got {other:?}"),
            }
        }
    }

    #[test]
    fn real_boundary_promotes_all_eight_integer_classes_to_double() {
        let inputs = [
            IntegerStorage::I8(vec![-1, 2]),
            IntegerStorage::I16(vec![-1, 2]),
            IntegerStorage::I32(vec![-1, 2]),
            IntegerStorage::I64(vec![-1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ];

        for input in inputs {
            let tensor = Tensor::new_integer(input, vec![2, 1]).expect("integer tensor");
            let output = map_real_tensor(tensor, "inverse-test", |value| value, |value| value)
                .expect("integer-to-double boundary");
            assert_eq!(output.numeric_dtype(), NumericDType::F64);
            assert_eq!(output.shape(), &[2, 1]);
        }
    }
}
