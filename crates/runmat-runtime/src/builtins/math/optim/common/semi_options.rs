//! Exact scalar controls shared by the semi-infinite solver and constructors.
use super::{optim_error, option_f64_value, option_usize_value};
use crate::BuiltinResult;
use runmat_builtins::{OptimDisplay, OptimOptionField as Field};
use runmat_value::Value;

pub(crate) fn display(value: &Value) -> BuiltinResult<(OptimDisplay, bool)> {
    if !matches!(value, Value::String(_) | Value::CharArray(_)) {
        return Err(optim_error("fseminf", "Display must be scalar text"));
    }
    let text = super::option_string_value(Some(value), "Display", "")?;
    let text = text.to_ascii_lowercase();
    let (base, detailed) = text
        .strip_suffix("-detailed")
        .map_or((text.as_str(), false), |base| (base, true));
    if detailed && !matches!(base, "iter" | "notify" | "final") {
        return Err(optim_error("fseminf", "unsupported detailed Display value"));
    }
    let mode = OptimDisplay::parse(base)
        .filter(|display| runmat_builtins::OptimSolver::Fseminf.accepts_display(*display))
        .ok_or_else(|| optim_error("fseminf", "unsupported Display value"))?;
    Ok((mode, detailed))
}
pub(crate) fn on_off(value: &Value, field: Field) -> BuiltinResult<bool> {
    if !matches!(value, Value::String(_) | Value::CharArray(_)) {
        return Err(optim_error(
            "fseminf",
            format!("{} must be on or off", field.name()),
        ));
    }
    match super::option_string_value(Some(value), field.name(), "")?
        .to_ascii_lowercase()
        .as_str()
    {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err(optim_error(
            "fseminf",
            format!("{} must be on or off", field.name()),
        )),
    }
}
pub(crate) fn number(value: &Value, field: Field) -> BuiltinResult<f64> {
    // DiffMaxChange alone documents an infinite upper step bound.
    if field == Field::DiffMaxChange {
        let infinite = match value {
            Value::Num(v) => *v == f64::INFINITY,
            Value::Tensor(v) if v.len() == 1 && v.integer_storage().is_none() => {
                crate::builtins::common::tensor::tensor_value_f64(v, 0) == f64::INFINITY
            }
            _ => false,
        };
        if infinite {
            return Ok(f64::INFINITY);
        }
    }
    option_f64_value("fseminf", Some(value), field.name(), 0.0)
}
pub(crate) fn validate(field: Field, value: &Value) -> BuiltinResult<()> {
    use Field::*;
    if !matches!(field, Diagnostics | FunValCheck) {
        super::preflight_option_scalar("fseminf", value, field)?;
    }
    match field {
        Diagnostics | FunValCheck => {
            on_off(value, field)?;
        }
        MaxSQPIter | RelLineSrchBndDuration => {
            if option_usize_value("fseminf", Some(value), field.name(), 0)? == 0 {
                return Err(optim_error(
                    "fseminf",
                    format!("{} must be positive", field.name()),
                ));
            }
        }
        DiffMaxChange | DiffMinChange | RelLineSrchBnd | TolConSQP => {
            let value = number(value, field)?;
            if value < 0.0 || (value == 0.0 && matches!(field, DiffMaxChange | TolConSQP)) {
                return Err(optim_error(
                    "fseminf",
                    format!("invalid {} range", field.name()),
                ));
            }
        }
        _ => return Err(optim_error("fseminf", "invalid semi-infinite option role")),
    }
    Ok(())
}
