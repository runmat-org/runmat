//! Shared typed network controls and stream configuration.

use runmat_value::{Tensor, Value};
use std::time::Duration;
use thiserror::Error;

pub(crate) const DEFAULT_TIMEOUT_SECONDS: f64 = 10.0;

#[derive(Debug, Error)]
pub(crate) enum TimeoutParseError {
    #[error("Timeout must be a scalar")]
    NonScalar,
    #[error("Timeout must be numeric")]
    NonNumeric,
    #[error("Timeout must be finite or Inf")]
    NonFinite,
    #[error("Timeout must be non-negative")]
    Negative,
    #[error("Timeout exceeds the supported host duration range")]
    OutOfRange,
}

pub(crate) fn parse_timeout_value(value: &Value) -> Result<f64, TimeoutParseError> {
    let timeout = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(t) if crate::builtins::common::tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                int.to_f64()
            } else {
                crate::builtins::common::tensor::tensor_value_f64(t, 0)
            }
        }
        Value::Tensor(_) => return Err(TimeoutParseError::NonScalar),
        _ => return Err(TimeoutParseError::NonNumeric),
    };
    if !timeout.is_finite() && !timeout.is_infinite() {
        return Err(TimeoutParseError::NonFinite);
    }
    if timeout.is_sign_negative() {
        return Err(TimeoutParseError::Negative);
    }
    if timeout.is_finite() && Duration::try_from_secs_f64(timeout).is_err() {
        return Err(TimeoutParseError::OutOfRange);
    }
    Ok(timeout)
}

#[derive(Debug, Error)]
#[error("{message}")]
pub(crate) struct StringScalarError {
    message: String,
}

impl StringScalarError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

pub(crate) fn string_scalar(value: &Value, context: &str) -> Result<String, StringScalarError> {
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        StringScalarError::new(format!(
            "{context} must be a string scalar or character vector (got {value:?})"
        ))
    })
}

#[derive(Debug, Error)]
#[error("{message}")]
pub(crate) struct PortParseError {
    message: String,
}

impl PortParseError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

pub(crate) fn parse_port(value: &Value) -> Result<u16, PortParseError> {
    let port = match value {
        Value::Int(int) => {
            let port = int
                .try_to_u64()
                .ok_or_else(|| PortParseError::new("port must be non-negative"))?;
            return u16::try_from(port).map_err(|_| {
                PortParseError::new(format!("port {port} is outside the valid range 0–65535"))
            });
        }
        Value::Num(num) => {
            if !num.is_finite() {
                return Err(PortParseError::new("port must be finite"));
            }
            if num.fract() != 0.0 {
                return Err(PortParseError::new("port must be an integer value"));
            }
            *num as i64
        }
        Value::Tensor(t) if crate::builtins::common::tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                let port = int
                    .try_to_u64()
                    .ok_or_else(|| PortParseError::new("port must be non-negative"))?;
                return u16::try_from(port).map_err(|_| {
                    PortParseError::new(format!("port {port} is outside the valid range 0–65535"))
                });
            }
            let raw = crate::builtins::common::tensor::tensor_value_f64(t, 0);
            if !raw.is_finite() {
                return Err(PortParseError::new("port must be finite"));
            }
            if raw.fract() != 0.0 {
                return Err(PortParseError::new("port must be an integer value"));
            }
            raw as i64
        }
        _ => return Err(PortParseError::new("port must be a numeric scalar")),
    };
    if !(0..=65_535).contains(&port) {
        return Err(PortParseError::new(format!(
            "port {port} is outside the valid range 0–65535"
        )));
    }
    Ok(port as u16)
}

pub(crate) fn canonicalize_byte_order(raw: &str) -> Option<&'static str> {
    let mut compact = String::with_capacity(raw.len());
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            compact.push(ch.to_ascii_lowercase());
        }
    }
    match compact.as_str() {
        "littleendian" | "little" => Some("little-endian"),
        "bigendian" | "big" => Some("big-endian"),
        _ => None,
    }
}

pub(crate) fn default_user_data() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}
