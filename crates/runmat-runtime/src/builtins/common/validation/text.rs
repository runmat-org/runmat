//! Text predicates, variable names, and name-value conversion.

use std::borrow::Cow;

use super::{
    class_name_for_value, invalid_argument_error, is_valid_varname, BuiltinResult, CellArray,
    CharArray, RuntimeError, Value,
};

fn is_character_row(chars: &CharArray) -> bool {
    chars.rows == 1 && crate::builtins::common::shape::effective_rank(&chars.shape) == 2
}

pub fn value_is_text(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) => true,
        Value::CharArray(chars) => is_character_row(chars),
        Value::Cell(cell) => cell.data().iter().all(value_is_text),
        _ => false,
    }
}

pub fn value_is_text_scalar(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::StringArray(strings) => {
            strings.data().len() == 1 && strings.rows == 1 && strings.cols == 1
        }
        Value::CharArray(chars) => is_character_row(chars),
        _ => false,
    }
}

/// Decode only the text-scalar forms admitted by `value_is_text_scalar`.
/// Generic `String::try_from(Value)` also stringifies numbers and booleans,
/// so it must not be used for a semantic text argument.
pub fn value_text_scalar_string(value: &Value) -> Option<String> {
    if !value_is_text_scalar(value) {
        return None;
    }
    match value {
        Value::String(text) => text.as_str().map(str::to_owned),
        Value::StringArray(strings) => strings
            .data()
            .first()
            .and_then(|text| text.as_str().map(str::to_owned)),
        Value::CharArray(chars) => chars.row_string(),
        _ => None,
    }
}

pub fn value_is_nonzero_length_text(value: &Value) -> bool {
    if !value_is_text(value) {
        return false;
    }
    match value {
        Value::String(s) => s.as_str().is_some_and(|text| !text.is_empty()),
        Value::StringArray(s) => s
            .data()
            .iter()
            .all(|value| value.as_str().is_some_and(|text| !text.is_empty())),
        Value::CharArray(c) => c.rows == 1 && c.cols > 0,
        Value::Cell(c) => c.data().iter().all(value_is_nonzero_length_text),
        _ => false,
    }
}

pub(super) fn value_texts(value: &Value) -> Result<Vec<Cow<'_, str>>, RuntimeError> {
    match value {
        Value::String(s) => s
            .as_str()
            .map(|text| vec![Cow::Borrowed(text)])
            .ok_or_else(|| {
                invalid_argument_error(
                    "argumentValidation",
                    "missing string is not a text argument",
                )
            }),
        Value::StringArray(s) => s
            .data()
            .iter()
            .map(|text| {
                text.as_str().map(Cow::Borrowed).ok_or_else(|| {
                    invalid_argument_error(
                        "argumentValidation",
                        "missing string is not a text argument",
                    )
                })
            })
            .collect(),
        Value::CharArray(c) if is_character_row(c) => Ok(vec![Cow::Owned(chars_to_string(c)?)]),
        Value::Cell(c) => {
            let mut out = Vec::with_capacity(c.data().len());
            for entry in c.data() {
                out.extend(value_texts(entry)?);
            }
            Ok(out)
        }
        other => Err(invalid_argument_error(
            "argumentValidation",
            format!("expected text, got {}", class_name_for_value(other)),
        )),
    }
}

pub(super) fn chars_to_string(chars: &CharArray) -> Result<String, RuntimeError> {
    chars
        .try_row_string()
        .map_err(|source| {
            crate::runtime_error::semantic_error_with_source(
                "InvalidArgument",
                "character text contains malformed UTF-16",
                source,
            )
        })?
        .ok_or_else(|| invalid_argument_error("argumentValidation", "expected character row"))
}

pub fn isvarname_value(value: &Value) -> bool {
    value_texts(value)
        .map(|names| names.iter().all(|name| is_valid_varname(name)))
        .unwrap_or(false)
}

pub fn namedargs2cell_value(value: Value) -> BuiltinResult<Value> {
    let Value::Struct(struct_value) = value else {
        return Err(
            invalid_argument_error("namedargs2cell", "input must be a scalar struct").into(),
        );
    };
    let mut data = Vec::with_capacity(struct_value.fields.len().saturating_mul(2));
    for (field, value) in struct_value.into_fields_unbounded() {
        data.push(Value::string(field));
        data.push(value);
    }
    let cols = data.len();
    let cell = CellArray::new(data, 1, cols)
        .map_err(|err| invalid_argument_error("namedargs2cell", err))?;
    Ok(Value::Cell(cell))
}

pub fn validate_function_signatures_json(value: &Value) -> BuiltinResult<()> {
    for text in value_texts(value)? {
        serde_json::from_str::<serde_json::Value>(&text).map_err(|err| {
            invalid_argument_error(
                "validateFunctionSignaturesJSON",
                format!("invalid JSON signature payload: {err}"),
            )
        })?;
    }
    Ok(())
}
