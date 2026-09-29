//! Explicit and deterministic default angular-frequency grids.

use super::*;
use crate::builtins::common::tensor;
use crate::builtins::control::tf_model::polynomial_roots;

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
        None => default_grid(model),
    }
}

fn values(value: &Value) -> BuiltinResult<Vec<f64>> {
    let data = match value {
        Value::Num(value) => vec![*value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Bool(value) => vec![if *value { 1.0 } else { 0.0 }],
        Value::Tensor(value) => {
            vector_shape(&value.shape)?;
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

fn default_grid(model: &TfModel) -> BuiltinResult<Vec<f64>> {
    if model.is_discrete() {
        let stop = std::f64::consts::PI / model.sample_time;
        if !stop.is_finite() {
            return Err(control_error(
                BUILTIN_NAME,
                "RunMat:nyquist:InvalidModel",
                "nyquist: sample time is too small for a finite default frequency grid",
            ));
        }
        return Ok((0..DEFAULT_FREQUENCY_POINTS)
            .map(|index| stop * (index + 1) as f64 / (DEFAULT_FREQUENCY_POINTS + 1) as f64)
            .collect());
    }
    let mut breakpoints = Vec::new();
    for coefficients in [&model.numerator, &model.denominator] {
        breakpoints.extend(
            polynomial_roots(coefficients, BUILTIN_NAME)?
                .into_iter()
                .map(|root| root.norm())
                .filter(|value| value.is_finite() && *value > 0.0),
        );
    }
    let (start, stop) = if breakpoints.is_empty() {
        (1.0e-2, 1.0e2)
    } else {
        let min = breakpoints.iter().copied().fold(f64::INFINITY, f64::min);
        let max = breakpoints.iter().copied().fold(0.0, f64::max);
        let start = (min / 100.0).max(1.0e-4);
        let stop = (max * 100.0).max(start * 10.0);
        (start, stop)
    };
    if !start.is_finite() || !stop.is_finite() {
        return Err(control_error(
            BUILTIN_NAME,
            "RunMat:nyquist:InvalidModel",
            "nyquist: model dynamics exceed a finite default frequency grid",
        ));
    }
    let first = start.log10();
    let span = stop.log10() - first;
    Ok((0..DEFAULT_FREQUENCY_POINTS)
        .map(|index| {
            10.0_f64.powf(first + span * index as f64 / (DEFAULT_FREQUENCY_POINTS - 1) as f64)
        })
        .collect())
}
