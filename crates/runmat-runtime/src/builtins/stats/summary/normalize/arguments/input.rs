//! Input admission and exact binary64 boundaries for numeric normalization.

use super::*;
use crate::gather_if_needed_async;
use runmat_value::{IntValue, NumericScalar};

pub(in crate::builtins::stats::summary::normalize) async fn value_to_input(
    value: Value,
) -> BuiltinResult<NumericInput> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(super::super::normalize_transfer_error)?;
    match value {
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(&tensor, "input data")?;
            let shape = normalize_shape_for(tensor.shape(), tensor::tensor_element_len(&tensor));
            Ok(NumericInput::Real {
                shape,
                data: tensor::tensor_into_values_f64(tensor),
            })
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| normalize_internal(format!("normalize: {err}")))?;
            let len = tensor.len();
            let shape = normalize_shape_for(tensor.shape(), len);
            Ok(NumericInput::Real {
                shape,
                data: tensor::tensor_into_values_f64(tensor),
            })
        }
        Value::Num(number) => Ok(NumericInput::Real {
            data: vec![number],
            shape: vec![1, 1],
        }),
        Value::Int(integer) => Ok(NumericInput::Real {
            data: vec![exact_integer_as_f64(&integer, "input data")?],
            shape: vec![1, 1],
        }),
        Value::Bool(bit) => Ok(NumericInput::Real {
            data: vec![if bit { 1.0 } else { 0.0 }],
            shape: vec![1, 1],
        }),
        Value::Complex(real, imaginary) => Ok(NumericInput::Complex {
            data: vec![(real, imaginary)],
            shape: vec![1, 1],
        }),
        Value::ComplexTensor(tensor) => Ok(NumericInput::Complex {
            shape: {
                ensure_exact_complex_integer_tensor(&tensor, "input data")?;
                normalize_shape_for(tensor.shape(), tensor::complex_tensor_element_len(&tensor))
            },
            data: tensor::complex_tensor_into_values_complex64(tensor)
                .into_iter()
                .map(|value| (value.re, value.im))
                .collect(),
        }),
        other => Err(normalize_error(format!(
            "normalize: unsupported input type {other:?}"
        ))),
    }
}

pub(super) fn ensure_exact_integer_value(value: &Value, role: &str) -> BuiltinResult<()> {
    match value {
        Value::Int(value) => {
            exact_integer_as_f64(value, role)?;
            Ok(())
        }
        Value::Tensor(tensor) => ensure_exact_integer_tensor(tensor, role),
        Value::ComplexTensor(tensor) => ensure_exact_complex_integer_tensor(tensor, role),
        _ => Ok(()),
    }
}

fn ensure_exact_complex_integer_tensor(tensor: &ComplexTensor, role: &str) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    for index in 0..tensor.len() {
        let Some((real, imag)) = tensor.numeric_value_at(index) else {
            continue;
        };
        if !numeric_scalar_is_exact_f64(real) || !numeric_scalar_is_exact_f64(imag) {
            return Err(normalize_error(format!(
                "normalize: complex integer {role} must be exactly representable as double"
            )));
        }
    }
    Ok(())
}

fn numeric_scalar_is_exact_f64(value: NumericScalar) -> bool {
    value
        .into_int_value()
        .is_none_or(|integer| crate::builtins::common::validation::integer_is_exact_f64(&integer))
}

fn exact_integer_as_f64(value: &IntValue, role: &str) -> BuiltinResult<f64> {
    if crate::builtins::common::validation::integer_is_exact_f64(value) {
        Ok(value.to_f64())
    } else {
        Err(normalize_error(format!(
            "normalize: integer {role} must be exactly representable as double"
        )))
    }
}

fn ensure_exact_integer_tensor(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    if let Some(storage) = tensor.integer_storage() {
        for integer in storage.exact_values() {
            if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
                return Err(normalize_error(format!(
                    "normalize: integer {role} must be exactly representable as double"
                )));
            }
        }
    }
    Ok(())
}
