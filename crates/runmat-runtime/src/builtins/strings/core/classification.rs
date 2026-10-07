//! Shared host character-property primitives for the text classifiers.

use once_cell::sync::Lazy;
use regex::Regex;
use runmat_builtins::CharacterPropertyCategory;
use runmat_value::{IntValue, NumericScalar, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::core::compat::{compat_error, logical_value, scalar_text};
use crate::{make_cell_with_shape, BuiltinResult};

static UNICODE_DECIMAL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\p{Nd}$").expect("valid Unicode decimal pattern"));
static UNICODE_PUNCTUATION_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\p{P}$").expect("valid Unicode punctuation pattern"));
static UNICODE_NON_GRAPHIC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?:\p{Zl}|\p{Zp}|\p{Co}|\p{Cn})$").expect("valid Unicode non-graphic pattern")
});

pub(crate) fn classify_text_value(
    value: Value,
    fn_name: &str,
    pred: impl Fn(char) -> bool + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            let data = text
                .chars()
                .map(|ch| u8::from(pred(ch)))
                .collect::<Vec<_>>();
            logical_value(data, vec![1, text.chars().count()], fn_name)
        }
        Value::StringArray(array) => {
            let values = array
                .data
                .into_iter()
                .map(|text| classify_text_value(Value::String(text), fn_name, pred))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, array.shape).map_err(|e| compat_error(fn_name, e))
        }
        Value::CharArray(array) => {
            let data = array
                .data
                .iter()
                .map(|ch| u8::from(pred(*ch)))
                .collect::<Vec<_>>();
            logical_value(data, vec![array.rows, array.cols], fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data
                .into_iter()
                .map(|value| classify_text_value(value, fn_name, pred))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape).map_err(|e| compat_error(fn_name, e))
        }
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

pub(crate) fn classify_text_or_numeric_value(
    value: Value,
    fn_name: &str,
    pred: impl Fn(char) -> bool + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => logical_value(
            vec![u8::from(floating_character_code(value).is_some_and(pred))],
            vec![1, 1],
            fn_name,
        ),
        Value::Int(value) => logical_value(
            vec![u8::from(integer_character_code(&value).is_some_and(pred))],
            vec![1, 1],
            fn_name,
        ),
        Value::Tensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let mut data = Vec::with_capacity(tensor_utils::tensor_element_len(&tensor));
            for index in 0..tensor_utils::tensor_element_len(&tensor) {
                let value = tensor.numeric_value_at(index).ok_or_else(|| {
                    compat_error(fn_name, format!("{fn_name}: inconsistent numeric storage"))
                })?;
                let character = match value {
                    NumericScalar::F64(value) => floating_character_code(value),
                    NumericScalar::F32(value) => floating_character_code(f64::from(value)),
                    value => integer_character_code(
                        &value
                            .into_int_value()
                            .expect("non-floating numeric scalar is integer"),
                    ),
                };
                data.push(u8::from(character.is_some_and(pred)));
            }
            logical_value(data, shape, fn_name)
        }
        other => classify_text_value(other, fn_name, pred),
    }
}

pub(crate) fn parse_force_cell_output(rest: &[Value]) -> BuiltinResult<bool> {
    if rest.is_empty() {
        return Ok(false);
    }
    if rest.len() != 2 {
        return Err(compat_error(
            "isstrprop",
            "isstrprop: expected the optional 'ForceCellOutput', tf name-value pair",
        ));
    }
    let option = scalar_text(&rest[0], "isstrprop")?;
    if !option.eq_ignore_ascii_case("forcecelloutput") {
        return Err(compat_error(
            "isstrprop",
            format!("isstrprop: unsupported option '{option}'"),
        ));
    }
    match &rest[1] {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 => Ok(false),
        Value::Num(value) if *value == 1.0 => Ok(true),
        Value::LogicalArray(value) if value.data.len() == 1 => Ok(value.data[0] != 0),
        _ => Err(compat_error(
            "isstrprop",
            "isstrprop: ForceCellOutput must be a logical scalar or numeric 0 or 1",
        )),
    }
}

fn floating_character_code(value: f64) -> Option<char> {
    let code = if value.is_finite() {
        value.round().clamp(0.0, u16::MAX as f64) as u32
    } else {
        0
    };
    char::from_u32(code)
}

fn integer_character_code(value: &IntValue) -> Option<char> {
    let code = match value {
        IntValue::I8(value) => i128::from(*value).clamp(0, i128::from(u16::MAX)) as u32,
        IntValue::I16(value) => i128::from(*value).clamp(0, i128::from(u16::MAX)) as u32,
        IntValue::I32(value) => i128::from(*value).clamp(0, i128::from(u16::MAX)) as u32,
        IntValue::I64(value) => i128::from(*value).clamp(0, i128::from(u16::MAX)) as u32,
        IntValue::U8(value) => u128::from(*value).min(u128::from(u16::MAX)) as u32,
        IntValue::U16(value) => u128::from(*value).min(u128::from(u16::MAX)) as u32,
        IntValue::U32(value) => u128::from(*value).min(u128::from(u16::MAX)) as u32,
        IntValue::U64(value) => u128::from(*value).min(u128::from(u16::MAX)) as u32,
    };
    char::from_u32(code)
}

pub(crate) fn char_matches_prop(ch: char, prop: CharacterPropertyCategory) -> bool {
    match prop {
        CharacterPropertyCategory::Alpha => ch.is_alphabetic(),
        CharacterPropertyCategory::Alphanum => ch.is_alphanumeric(),
        CharacterPropertyCategory::Digit => {
            UNICODE_DECIMAL_RE.is_match(ch.encode_utf8(&mut [0; 4]))
        }
        CharacterPropertyCategory::XDigit => ch.is_ascii_hexdigit(),
        CharacterPropertyCategory::WSpace => ch.is_whitespace(),
        CharacterPropertyCategory::Upper => ch.is_uppercase(),
        CharacterPropertyCategory::Lower => ch.is_lowercase(),
        CharacterPropertyCategory::Punct => {
            UNICODE_PUNCTUATION_RE.is_match(ch.encode_utf8(&mut [0; 4]))
        }
        CharacterPropertyCategory::Control => ch.is_control(),
        CharacterPropertyCategory::Graphic => {
            !ch.is_control()
                && !ch.is_whitespace()
                && !UNICODE_NON_GRAPHIC_RE.is_match(ch.encode_utf8(&mut [0; 4]))
        }
        CharacterPropertyCategory::Print => {
            ch == ' '
                || (!ch.is_control()
                    && !ch.is_whitespace()
                    && !UNICODE_NON_GRAPHIC_RE.is_match(ch.encode_utf8(&mut [0; 4])))
        }
    }
}
