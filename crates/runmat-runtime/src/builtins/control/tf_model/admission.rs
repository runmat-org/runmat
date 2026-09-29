//! Admission of consumed host model properties before provider materialization.
use super::{
    all_zero, coefficients_from_property, control_error, invalid_model_identifier, scalar_f64,
    scalar_property, scalar_text, unsupported_model_identifier, validate_coefficients,
    validate_delay, validate_sample_time, validate_variable, validate_variable_domain,
};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::{ObjectInstance, Value};

pub(super) fn preflight_host_model_properties(
    object: &ObjectInstance,
    builtin: &'static str,
) -> BuiltinResult<()> {
    for (name, label) in [("Numerator", "numerator"), ("Denominator", "denominator")] {
        if !matches!(object.properties.get(name), Some(Value::GpuTensor(_))) {
            let coefficients = coefficients_from_property(object, name, builtin)?;
            validate_coefficients(label, &coefficients, builtin)?;
            if name == "Denominator" && all_zero(&coefficients) {
                return Err(control_error(
                    builtin,
                    invalid_model_identifier(builtin),
                    format!("{builtin}: denominator coefficients must not all be zero"),
                ));
            }
        }
    }
    for name in ["Ts", "InputDelay", "OutputDelay"] {
        if !matches!(object.properties.get(name), Some(Value::GpuTensor(_))) {
            let value = scalar_property(object, name, builtin)?;
            if name == "Ts" {
                validate_sample_time(value, builtin)?;
            } else {
                validate_delay(value, name, builtin)?;
            }
        }
    }
    let variable = object
        .properties
        .get("Variable")
        .filter(|value| !matches!(value, Value::GpuTensor(_)))
        .map(|value| {
            scalar_text(value, "Variable", builtin)
                .and_then(|text| validate_variable(&text, builtin))
        })
        .transpose()?;
    if let (Some(variable), Some(sample_time)) = (variable, object.properties.get("Ts")) {
        if !matches!(sample_time, Value::GpuTensor(_)) {
            validate_variable_domain(&variable, scalar_property(object, "Ts", builtin)?, builtin)?;
        }
    }
    Ok(())
}

/// Check only the state-space fields consumed by an operation; unused model
/// properties must not affect admission or cause provider access.
pub fn preflight_ss_host_properties(
    object: &ObjectInstance,
    required: &[&str],
    builtin: &'static str,
) -> BuiltinResult<()> {
    for &name in required {
        let value = object.properties.get(name).ok_or_else(|| {
            control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: ss object is missing {name}"),
            )
        })?;
        if matches!(value, Value::GpuTensor(_)) {
            continue;
        }
        match name {
            "Ts" => validate_sample_time(scalar_f64(value, name, builtin)?, builtin)?,
            "InputDelay" | "OutputDelay" => {
                validate_delay(scalar_f64(value, name, builtin)?, name, builtin)?;
            }
            "A" => preflight_ss_state_matrix(value, builtin)?,
            "B" | "C" | "D" => preflight_ss_matrix_value(value, name, builtin)?,
            _ => {}
        }
    }
    Ok(())
}

fn preflight_ss_state_matrix(value: &Value, builtin: &'static str) -> BuiltinResult<()> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.shape.len() > 2 || tensor.rows != tensor.cols {
                return Err(control_error(
                    builtin,
                    invalid_model_identifier(builtin),
                    format!("{builtin}: ss A must be square, got {:?}", tensor.shape),
                ));
            }
            if tensor::tensor_values_f64_cow(tensor)
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(control_error(
                    builtin,
                    unsupported_model_identifier(builtin),
                    format!("{builtin}: ss A must contain only finite real values"),
                ));
            }
        }
        Value::Num(value) if value.is_finite() => {}
        Value::Int(_) => {}
        other => {
            return Err(control_error(
                builtin,
                unsupported_model_identifier(builtin),
                format!("{builtin}: ss A must be a finite real matrix, got {other:?}"),
            ));
        }
    }
    Ok(())
}

fn preflight_ss_matrix_value(
    value: &Value,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let invalid_shape = |shape: &[usize]| {
        control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: ss {name} must be a 2-D matrix, got {shape:?}"),
        )
    };
    let finite = match value {
        Value::Tensor(tensor) => {
            if tensor.shape.len() > 2 {
                return Err(invalid_shape(&tensor.shape));
            }
            tensor::tensor_values_f64_cow(tensor)
                .iter()
                .all(|value| value.is_finite())
        }
        Value::ComplexTensor(tensor) => {
            if tensor.shape.len() > 2 {
                return Err(invalid_shape(&tensor.shape));
            }
            tensor::complex_tensor_values_complex64(tensor)
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        }
        Value::LogicalArray(logical) => {
            if logical.shape.len() > 2 {
                return Err(invalid_shape(&logical.shape));
            }
            true
        }
        Value::Num(value) => value.is_finite(),
        Value::Int(_) | Value::Bool(_) => true,
        Value::Complex(re, im) => re.is_finite() && im.is_finite(),
        other => {
            return Err(control_error(
                builtin,
                unsupported_model_identifier(builtin),
                format!("{builtin}: ss {name} must be a finite numeric matrix, got {other:?}"),
            ));
        }
    };
    if !finite {
        return Err(control_error(
            builtin,
            unsupported_model_identifier(builtin),
            format!("{builtin}: ss {name} must contain finite numeric values"),
        ));
    }
    Ok(())
}
