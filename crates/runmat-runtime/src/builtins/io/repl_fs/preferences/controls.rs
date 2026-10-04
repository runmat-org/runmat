use runmat_value::{CharArray, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) fn error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub(super) fn map_control_flow(name: &str, source: RuntimeError) -> RuntimeError {
    let identifier = source.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{name}: {}", source.message()))
        .with_builtin(name)
        .with_source(source);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(char_row_to_string(array)),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(error(
            name,
            format!("{name}: {arg} must be a string scalar or character vector"),
        )),
    }
}

pub(super) fn char_row_to_string(array: &CharArray) -> String {
    let mut text: String = array.data.iter().take(array.cols).collect();
    while text.ends_with(' ') {
        text.pop();
    }
    text
}

#[derive(Debug)]
pub(super) enum PreferenceNames {
    Scalar(String),
    Array {
        names: Vec<String>,
        shape: Vec<usize>,
    },
}

pub(super) fn preference_names(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<PreferenceNames> {
    match value {
        Value::String(text) => Ok(PreferenceNames::Scalar(text.clone())),
        Value::CharArray(array) if array.rows == 1 => {
            Ok(PreferenceNames::Scalar(char_row_to_string(array)))
        }
        Value::StringArray(array) if array.data.len() == 1 => {
            Ok(PreferenceNames::Scalar(array.data[0].clone()))
        }
        Value::StringArray(array) => Ok(PreferenceNames::Array {
            names: array.data.clone(),
            shape: array.shape.clone(),
        }),
        Value::Cell(cell) => {
            let mut names = Vec::with_capacity(cell.data.len());
            for value in &cell.data {
                match value {
                    Value::CharArray(array) if array.rows == 1 => {
                        names.push(char_row_to_string(array));
                    }
                    _ => {
                        return Err(error(
                            builtin,
                            format!("{builtin}: preference cell entries must be character vectors"),
                        ));
                    }
                }
            }
            Ok(PreferenceNames::Array {
                names,
                shape: cell.shape.clone(),
            })
        }
        _ => Err(error(
            builtin,
            format!("{builtin}: preference must be a character vector, string scalar, cell array of character vectors, or string array"),
        )),
    }
}
