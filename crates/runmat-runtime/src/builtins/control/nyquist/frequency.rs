//! Explicit and deterministic default angular-frequency grids.

use super::*;
use crate::builtins::common::tensor;
use crate::builtins::control::frequency_response;

pub(super) fn validate_host(value: &Value) -> BuiltinResult<()> {
    let _ = values(value)?;
    Ok(())
}

pub(super) async fn parse(model: &TfModel, value: Option<&Value>) -> BuiltinResult<Vec<f64>> {
    match value {
        Some(value) => {
            let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
            super::admission::exact_gathered_frequency(&gathered)?;
            values(&gathered)
        }
        None => frequency_response::default_grid(frequency_response::FrequencyApi::Nyquist, model),
    }
}

fn values(value: &Value) -> BuiltinResult<Vec<f64>> {
    let data = match value {
        Value::Num(value) => vec![*value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Bool(value) => vec![if *value { 1.0 } else { 0.0 }],
        Value::Tensor(value) => {
            vector_shape(value.shape())?;
            tensor::tensor_values_f64(value)
        }
        Value::LogicalArray(value) => {
            vector_shape(&value.shape)?;
            value
                .data
                .iter()
                .map(|bit| if *bit == 0 { 0.0 } else { 1.0 })
                .collect()
        }
        _ => {
            return Err(control_error(
                BUILTIN_NAME,
                "RunMat:nyquist:InvalidArgument",
                "nyquist: frequency input must be a real numeric vector",
            ))
        }
    };
    if data.is_empty() || data.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(control_error(
            BUILTIN_NAME,
            "RunMat:nyquist:InvalidArgument",
            "nyquist: frequencies must be nonempty, finite and nonnegative",
        ));
    }
    Ok(data)
}

fn vector_shape(shape: &[usize]) -> BuiltinResult<()> {
    if shape.iter().filter(|dim| **dim > 1).count() > 1 {
        Err(control_error(
            BUILTIN_NAME,
            "RunMat:nyquist:InvalidArgument",
            "nyquist: frequency input must be a vector",
        ))
    } else {
        Ok(())
    }
}
