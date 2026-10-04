//! Shared typed network controls and stream configuration.

use runmat_value::Value;
use std::io;
use std::net::TcpStream;
use std::time::Duration;
use thiserror::Error;

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

pub(crate) fn configure_stream(stream: &TcpStream, timeout: f64) -> io::Result<()> {
    let opt = if timeout.is_infinite() || timeout == 0.0 {
        None
    } else {
        Some(Duration::from_secs_f64(timeout))
    };
    stream.set_read_timeout(opt)?;
    stream.set_write_timeout(opt)?;
    Ok(())
}
