use crate::builtins::common::tensor;
use runmat_value::Value;

pub(super) fn parse(value: Option<&Value>) -> crate::BuiltinResult<usize> {
    let Some(value) = value else { return Ok(1) };
    if let Some(value) = tensor::scalar_integer_value(value) {
        return value
            .try_to_usize()
            .filter(|value| *value >= 1)
            .ok_or_else(|| super::error::invalid("dim must be a positive integer"));
    }
    let Value::Num(raw) = value else {
        return Err(super::error::invalid("dim must be a positive integer"));
    };
    if !raw.is_finite() || *raw < 1.0 || raw.fract() != 0.0 {
        return Err(super::error::invalid("dim must be a positive integer"));
    }
    if *raw > usize::MAX as f64 || (usize::BITS == 64 && *raw == usize::MAX as f64) {
        return Err(super::error::invalid("dim exceeds platform limits"));
    }
    Ok(*raw as usize)
}
