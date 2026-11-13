//! Shared host character-property primitives for the text classifiers.

use once_cell::sync::Lazy;
use regex::Regex;
use runmat_builtins::CharacterPropertyCategory;
use runmat_value::{IntValue, NumericScalar, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::common::{scalar_text_input, text_input_admission};
use crate::builtins::strings::core::compat::{compat_error, logical_value};
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
    classify_text_ref(&value, fn_name, pred)
}

fn classify_text_ref(
    value: &Value,
    fn_name: &str,
    pred: impl Fn(char) -> bool + Copy,
) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => classify_string_element(text, fn_name, pred),
        Value::StringArray(array) => {
            let values = array
                .data()
                .iter()
                .map(|text| classify_string_element(text, fn_name, pred))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, array.shape().to_vec())
                .map_err(|e| compat_error(fn_name, e))
        }
        Value::CharArray(array) => {
            let data = array
                .data()
                .iter()
                // Char arrays expose UTF-16 storage geometry. A surrogate is
                // not a standalone Unicode scalar and cannot satisfy a property.
                .map(|unit| u8::from(char::from_u32(u32::from(unit.raw())).is_some_and(pred)))
                .collect::<Vec<_>>();
            logical_value(data, vec![array.rows, array.cols], fn_name)
        }
        Value::Cell(cell) => {
            let values = cell
                .data()
                .iter()
                .map(|value| classify_text_ref(value, fn_name, pred))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(values, cell.shape().to_vec())
                .map_err(|e| compat_error(fn_name, e))
        }
        other => Err(compat_error(
            fn_name,
            format!("{fn_name}: expected text input, got {other:?}"),
        )),
    }
}

pub(crate) fn classify_string(
    text: &str,
    fn_name: &str,
    pred: impl Fn(char) -> bool + Copy,
) -> BuiltinResult<Value> {
    let data = text
        .chars()
        .map(|ch| u8::from(pred(ch)))
        .collect::<Vec<_>>();
    logical_value(data, vec![1, text.chars().count()], fn_name)
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

pub(crate) fn parse_force_cell_output(
    runtime: Option<&crate::context::RuntimeContext>,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<bool> {
    if rest.is_empty() {
        return Ok(false);
    }
    if rest.len() != 2 {
        return Err(compat_error(
            "isstrprop",
            "isstrprop: expected the optional 'ForceCellOutput', tf name-value pair",
        ));
    }
    let option = scalar_text_input(&rest[0], runtime, |runtime, cause| {
        text_input_admission("isstrprop", runtime, cause)
    })?
    .ok_or_else(|| compat_error("isstrprop", "isstrprop: expected a text option"))?;
    if !option.as_str().eq_ignore_ascii_case("forcecelloutput") {
        return Err(compat_error(
            "isstrprop",
            format!("isstrprop: unsupported option '{}'", option.as_str()),
        ));
    }
    let flag = match &rest[1] {
        Value::Bool(value) => Some(*value),
        Value::Num(value) => double_binary_flag(*value),
        // A dense Double with one stored element has the same scalar flag
        // meaning as Value::Num. Borrow its native F64 storage: Single and
        // integer buffers must not enter through a converting read.
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .as_f64_slice()
            .and_then(|values| values.first().copied())
            .and_then(double_binary_flag),
        Value::LogicalArray(value) if value.data.len() == 1 => Some(value.data[0] != 0),
        _ => None,
    };
    flag.ok_or_else(|| {
        compat_error(
            "isstrprop",
            "isstrprop: ForceCellOutput must be a logical scalar or real Double 0 or 1",
        )
    })
}

fn double_binary_flag(value: f64) -> Option<bool> {
    if value == 0.0 {
        Some(false)
    } else if value == 1.0 {
        Some(true)
    } else {
        None
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

pub(crate) fn classify_string_element(
    text: &runmat_value::StringElement,
    fn_name: &str,
    pred: impl Fn(char) -> bool + Copy,
) -> BuiltinResult<Value> {
    match text.as_str() {
        Some(text) => classify_string(text, fn_name, pred),
        None => logical_value(vec![0], vec![1, 1], fn_name),
    }
}
