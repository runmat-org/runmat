//! Text predicates, variable names, and name-value conversion.

use super::{
    class_name_for_value, invalid_argument_error, is_valid_varname, BuiltinResult, CellArray,
    CharArray, RuntimeError, Value,
};

pub fn value_is_text(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) => true,
        Value::CharArray(chars) => chars.rows == 1,
        Value::Cell(cell) => cell.data.iter().all(value_is_text),
        _ => false,
    }
}

pub fn value_is_text_scalar(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::StringArray(strings) => {
            strings.data.len() == 1 && strings.rows == 1 && strings.cols == 1
        }
        Value::CharArray(chars) => chars.rows == 1,
        _ => false,
    }
}

pub fn value_is_nonzero_length_text(value: &Value) -> bool {
    if !value_is_text(value) {
        return false;
    }
    match value {
        Value::String(s) => !s.is_empty(),
        Value::StringArray(s) => s.data.iter().all(|value| !value.is_empty()),
        Value::CharArray(c) => c.rows == 1 && c.cols > 0,
        Value::Cell(c) => c.data.iter().all(value_is_nonzero_length_text),
        _ => false,
    }
}

pub(super) fn value_texts(value: &Value) -> Result<Vec<String>, RuntimeError> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::StringArray(s) => Ok(s.data.clone()),
        Value::CharArray(c) if c.rows == 1 => Ok(vec![chars_to_string(c)]),
        Value::Cell(c) => {
            let mut out = Vec::with_capacity(c.data.len());
            for entry in &c.data {
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

pub(super) fn chars_to_string(chars: &CharArray) -> String {
    chars.data.iter().collect()
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
    for (field, value) in struct_value.fields {
        data.push(Value::String(field));
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
