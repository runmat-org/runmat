//! Character and cellstr conversion to string values.

use runmat_macros::runtime_builtin;
use runmat_value::{StringArray, Value};

use crate::builtins::strings::common::char_row_to_string_slice;
use crate::builtins::strings::core::conversion::convert_variadic;
use crate::BuiltinResult;

const NAME: &str = "convertCharsToStrings";

#[runtime_builtin(
    name = "convertCharsToStrings",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::convert_chars_to_strings"
)]
pub(crate) async fn convert_chars_to_strings_builtin(
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    convert_variadic(value, rest, convert_value)
}

fn convert_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::CharArray(array) => {
            let mut text = String::with_capacity(array.data.len());
            for col in 0..array.cols {
                for row in 0..array.rows {
                    text.push(array.data[row * array.cols + col]);
                }
            }
            Ok(Value::String(text))
        }
        Value::Cell(cell) => {
            if !cell.data.iter().all(is_character_vector) {
                return Ok(Value::Cell(cell));
            }
            let data = cell
                .data
                .into_iter()
                .map(|value| match value {
                    Value::CharArray(array) if array.rows == 0 => Ok(String::new()),
                    Value::CharArray(array) => {
                        Ok(char_row_to_string_slice(&array.data, array.cols, 0))
                    }
                    _ => Err(conversion_error(
                        "convertCharsToStrings: invalid cellstr element",
                    )),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            StringArray::new(data, cell.shape)
                .map(Value::StringArray)
                .map_err(conversion_error)
        }
        other => Ok(other),
    }
}

fn is_character_vector(value: &Value) -> bool {
    matches!(value, Value::CharArray(array) if array.rows <= 1)
}

fn conversion_error(message: impl Into<String>) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(NAME)
        .build()
}
