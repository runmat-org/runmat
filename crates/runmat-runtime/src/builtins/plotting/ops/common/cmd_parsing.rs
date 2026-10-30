use runmat_value::{IntValue, Value};

use crate::builtins::common::tensor;
use crate::builtins::plotting::plotting_error;
use crate::BuiltinResult;

pub fn as_lower_str(val: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    Ok(match val {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.to_ascii_lowercase()),
        Value::CharArray(c)
            if c.rows <= 1 && crate::builtins::common::shape::effective_rank(&c.shape) <= 2 =>
        {
            Some(runmat_value::decode_utf16_code_units(c.data())?.to_ascii_lowercase())
        }
        _ => None,
    })
}

pub fn parse_on_off(
    builtin: &'static str,
    arg: Option<&Value>,
) -> Result<Option<bool>, crate::RuntimeError> {
    let Some(arg) = arg else {
        return Ok(None);
    };
    let Some(s) = as_lower_str(arg).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source(builtin, source.to_string(), source)
    })?
    else {
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
    if let Some(value) = tensor::scalar_integer_value(value) {
        return match value.try_to_i64() {
            Some(0) => Ok(HoldMode::Off),
            Some(1) => Ok(HoldMode::On),
            _ => Err(plotting_error(
                "hold",
                "hold: numeric state must be exactly 0 or 1",
            )),
        };
    }
    match value {
        Value::CharArray(chars) => {
            let text: String =
                runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "hold",
                        source.to_string(),
                        source,
                    )
                })?;
            parse_hold_mode_str(text.trim())
        }
        Value::String(runmat_value::StringElement::Text(s)) => parse_hold_mode_str(s.trim()),
        Value::Num(0.0) => Ok(HoldMode::Off),
        Value::Num(1.0) => Ok(HoldMode::On),
        Value::Num(_) => Err(plotting_error(
            "hold",
            "hold: numeric state must be exactly 0 or 1",
        )),
        Value::Bool(b) => Ok(if *b { HoldMode::On } else { HoldMode::Off }),
        Value::Tensor(tensor) => {
            if !tensor::is_scalar_tensor(tensor) {
                return Err(plotting_error("hold", "hold: logical scalar expected"));
            }
            match tensor::tensor_value_f64(tensor, 0) {
                0.0 => Ok(HoldMode::Off),
                1.0 => Ok(HoldMode::On),
                _ => Err(plotting_error(
                    "hold",
                    "hold: numeric state must be exactly 0 or 1",
                )),
            }
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
    fn mode_text_rejects_character_pages_and_rows_without_changing_empty_policy() {
        for shape in [vec![2, 2], vec![1, 2, 2], vec![1, 1, 4]] {
            let chars = runmat_value::CharArray::from_unicode_scalars_with_shape(
                "flat".chars().collect(),
                shape,
            )
            .unwrap();
            assert_eq!(as_lower_str(&Value::CharArray(chars)).unwrap(), None);
        }
        for shape in [vec![1, 4], vec![1, 4, 1, 1]] {
            let chars = runmat_value::CharArray::from_unicode_scalars_with_shape(
                "FLAT".chars().collect(),
                shape,
            )
            .unwrap();
            assert_eq!(
                as_lower_str(&Value::CharArray(chars)).unwrap(),
                Some("flat".into())
            );
        }
        let empty =
            runmat_value::CharArray::from_unicode_scalars_with_shape(vec![], vec![0, 0]).unwrap();
        assert_eq!(
            as_lower_str(&Value::CharArray(empty)).unwrap(),
            Some(String::new())
        );
    }

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
    fn hold_parser_rejects_numeric_values_outside_documented_state_domain() {
        for value in [Value::Num(-1.0), Value::Num(2.0), Value::Num(f64::NAN)] {
            assert!(parse_hold_mode(&value).is_err());
        }
        let typed = Value::Int(IntValue::U64(u64::MAX));
        assert!(parse_hold_mode(&typed).is_err());
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
