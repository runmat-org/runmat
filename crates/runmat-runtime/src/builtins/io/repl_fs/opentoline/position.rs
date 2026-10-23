use super::*;
use crate::{builtins::common::tensor, gather_if_needed_async};
pub(super) enum AdmittedPosition {
    Host(usize),
    Resident(Value),
}
pub(super) fn admit_position(value: &Value, label: &str) -> BuiltinResult<AdmittedPosition> {
    if let Value::GpuTensor(handle) = value {
        if handle
            .shape()
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            != Some(1)
        {
            return Err(opentoline_error(
                &OPENTOLINE_ERROR_POSITION,
                format!("opentoline: {label} must be a scalar"),
            ));
        }
        return Ok(AdmittedPosition::Resident(value.clone()));
    }
    host_position(value, label).map(AdmittedPosition::Host)
}
pub(super) async fn resolve_position(value: AdmittedPosition, label: &str) -> BuiltinResult<usize> {
    match value {
        AdmittedPosition::Host(position) => Ok(position),
        AdmittedPosition::Resident(value) => {
            let gathered = gather_if_needed_async(&value)
                .await
                .map_err(|err| opentoline_flow_error("opentoline", err))?;
            host_position(&gathered, label)
        }
    }
}
pub(super) fn host_position(value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Some(text) = super::super::file_dialog::try_scalar_text(value, label, |detail| {
        opentoline_error(&OPENTOLINE_ERROR_POSITION, detail)
    })? {
        return parse_positive_integer_text(&text, label);
    }
    if let Value::Int(integer) = value {
        return integer
            .try_to_usize()
            .filter(|position| *position > 0)
            .ok_or_else(|| {
                opentoline_error(
                    &OPENTOLINE_ERROR_POSITION,
                    format!("opentoline: {label} must be a positive integer scalar"),
                )
            });
    }
    let Some(position) = (match value {
        Value::Num(value) => position_value_to_usize(*value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            position_tensor_to_usize(tensor)
        }
        _ => {
            return Err(opentoline_error(
                &OPENTOLINE_ERROR_POSITION,
                format!("opentoline: {label} must be a numeric scalar"),
            ))
        }
    }) else {
        return Err(opentoline_error(
            &OPENTOLINE_ERROR_POSITION,
            format!("opentoline: {label} must be a positive integer scalar"),
        ));
    };
    Ok(position)
}

fn parse_positive_integer_text(text: &str, label: &str) -> BuiltinResult<usize> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(opentoline_error(
            &OPENTOLINE_ERROR_POSITION,
            format!("opentoline: {label} must not be empty"),
        ));
    }
    let value = trimmed.parse::<usize>().map_err(|_| {
        opentoline_error(
            &OPENTOLINE_ERROR_POSITION,
            format!("opentoline: {label} must be a positive integer scalar"),
        )
    })?;
    if value == 0 {
        return Err(opentoline_error(
            &OPENTOLINE_ERROR_POSITION,
            format!("opentoline: {label} must be positive"),
        ));
    }
    Ok(value)
}

fn position_value_to_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 1.0 || value.fract().abs() > f64::EPSILON {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

fn position_tensor_to_usize(tensor: &Tensor) -> Option<usize> {
    if let Some(storage) = tensor.integer_storage() {
        return storage
            .value_at(0)
            .and_then(|value| value.try_to_usize())
            .filter(|position| *position > 0);
    }
    position_value_to_usize(tensor::tensor_value_f64(tensor, 0))
}
