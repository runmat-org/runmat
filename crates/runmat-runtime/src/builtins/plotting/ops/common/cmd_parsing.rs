use runmat_value::{IntValue, Value};

use crate::builtins::common::tensor;
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
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return positive_index_from_integer(&integer, name);
    }
    match value {
        Value::Num(v) => to_positive_index(*v, name),
        Value::Bool(flag) => to_positive_index(if *flag { 1.0 } else { 0.0 }, name),
        Value::Tensor(tensor) => {
            if !tensor::is_scalar_tensor(tensor) {
                return Err(plotting_error(
                    name,
                    format!("{name}: expected scalar input"),
                ));
            }
            to_positive_index(tensor::tensor_values_f64(tensor)[0], name)
        }
        _ => Err(plotting_error(
            name,
            format!("{name}: unsupported argument type"),
        )),
    }
}

fn positive_index_from_integer(value: &IntValue, name: &str) -> BuiltinResult<usize> {
    let Some(index) = value.try_to_usize() else {
        return Err(plotting_error(
            name,
            format!("{name}: value must be a positive platform integer"),
        ));
    };
    if index == 0 {
        return Err(plotting_error(
            name,
            format!("{name}: value must be positive"),
        ));
    }
    Ok(index)
}

pub fn to_positive_index(value: f64, name: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(plotting_error(
            name,
            format!("{name}: value must be finite"),
        ));
    }
    let rounded = value.round();
    if rounded <= 0.0 {
        return Err(plotting_error(
            name,
            format!("{name}: value must be positive"),
        ));
    }
    if (rounded - value).abs() > f64::EPSILON
        || rounded > usize::MAX as f64
        || (usize::BITS == 64 && rounded == usize::MAX as f64)
    {
        return Err(plotting_error(
            name,
            format!("{name}: value must be a positive platform integer"),
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
            if !tensor::is_scalar_tensor(tensor) {
                return Err(plotting_error("hold", "hold: logical scalar expected"));
            }
            Ok(if tensor::tensor_values_f64(tensor)[0] == 0.0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn scalar_and_hold_parsers_read_typed_integer_storage() {
        let scalar_tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![4]), vec![1, 1]).expect("scalar");
        let scalar = Value::Tensor(scalar_tensor);
        let hold_tensor =
            Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("hold");
        let hold = Value::Tensor(hold_tensor);

        assert_eq!(scalar_from_value(&scalar, "subplot").expect("scalar"), 4);
        assert!(matches!(
            parse_hold_mode(&hold).expect("hold"),
            crate::builtins::plotting::state::HoldMode::On
        ));
    }

    #[test]
    fn scalar_parser_rejects_float_boundary_before_cast() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };

        assert!(scalar_from_value(&Value::Num(boundary), "subplot").is_err());
        assert!(scalar_from_value(&Value::Num(1.5), "subplot").is_err());
    }
}
