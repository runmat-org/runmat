use runmat_builtins::Value;

use crate::builtins::plotting::plotting_error;
use crate::BuiltinResult;

pub fn as_lower_str(val: &Value) -> Option<String> {
    match val {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::CharArray(c) => Some(c.data.iter().collect::<String>().to_ascii_lowercase()),
        _ => None,
    }
}

pub fn parse_on_off(
    builtin: &'static str,
    arg: Option<&Value>,
) -> Result<Option<bool>, crate::RuntimeError> {
    let Some(arg) = arg else {
        return Ok(None);
    };
    let Some(s) = as_lower_str(arg) else {
        return Err(plotting_error(builtin, "expected string argument"));
    };
    match s.trim() {
        "on" => Ok(Some(true)),
        "off" => Ok(Some(false)),
        other => Err(plotting_error(
            builtin,
            format!("expected 'on' or 'off' (got '{other}')"),
        )),
    }
}

pub fn scalar_from_value(value: &Value, name: &str) -> BuiltinResult<usize> {
    match value {
        Value::Num(v) => to_positive_index(*v, name),
        Value::Bool(flag) => to_positive_index(if *flag { 1.0 } else { 0.0 }, name),
        Value::Int(i) => to_positive_index(i.to_f64(), name),
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(plotting_error(
                    name,
                    format!("{name}: expected scalar input"),
                ));
            }
            to_positive_index(tensor.data[0], name)
        }
        _ => Err(plotting_error(
            name,
            format!("{name}: unsupported argument type"),
        )),
    }
}

pub fn to_positive_index(value: f64, name: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(plotting_error(
            name,
            format!("{name}: value must be finite"),
        ));
    }
    let rounded = value.round() as i64;
    if rounded <= 0 {
        return Err(plotting_error(
            name,
            format!("{name}: value must be positive"),
        ));
    }
    Ok(rounded as usize)
}

pub fn parse_hold_mode(value: &Value) -> BuiltinResult<crate::builtins::plotting::state::HoldMode> {
    use crate::builtins::plotting::state::HoldMode;
    match value {
        Value::CharArray(chars) => {
            let text: String = chars.data.iter().collect();
            parse_hold_mode_str(text.trim())
        }
        Value::String(s) => parse_hold_mode_str(s.trim()),
        Value::Num(v) => Ok(if *v == 0.0 {
            HoldMode::Off
        } else {
            HoldMode::On
        }),
        Value::Bool(b) => Ok(if *b { HoldMode::On } else { HoldMode::Off }),
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(plotting_error("hold", "hold: logical scalar expected"));
            }
            Ok(if tensor.data[0] == 0.0 {
                HoldMode::Off
            } else {
                HoldMode::On
            })
        }
        _ => Err(plotting_error("hold", "hold: unsupported argument type")),
    }
}

pub fn parse_hold_mode_str(
    text: &str,
) -> BuiltinResult<crate::builtins::plotting::state::HoldMode> {
    use crate::builtins::plotting::state::HoldMode;
    match text.to_ascii_lowercase().as_str() {
        "on" | "all" => Ok(HoldMode::On),
        "off" => Ok(HoldMode::Off),
        "" => Ok(HoldMode::Toggle),
        _ => Err(plotting_error("hold", "hold: expected 'on' or 'off'")),
    }
}
