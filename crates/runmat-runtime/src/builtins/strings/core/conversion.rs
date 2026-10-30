//! Shared mechanics for the positional text-conversion builtins.

use runmat_value::{CharArray, Value};

use crate::BuiltinResult;

pub(crate) fn convert_variadic(
    value: Value,
    rest: crate::call::BuiltinArguments,
    convert: fn(Value) -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    // All inputs are evaluated and converted before selecting requested outputs.
    let outputs = std::iter::once(value)
        .chain(rest)
        .map(convert)
        .collect::<BuiltinResult<Vec<_>>>()?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(n) => Ok(crate::output_count::output_list_with_padding(n, outputs)),
        None => Ok(outputs
            .into_iter()
            .next()
            .unwrap_or_else(|| Value::string(String::new()))),
    }
}

pub(crate) fn string_scalar_to_chars(text: &runmat_value::StringElement) -> CharArray {
    // convertStringsToChars represents a missing string as an empty character array.
    if text.as_str().is_none_or(str::is_empty) {
        CharArray::from_code_units(Vec::new(), 0, 0).expect("valid empty character array")
    } else {
        CharArray::new_row(text.as_str().expect("nonempty text branch"))
    }
}

pub(crate) fn string_to_chars(value: Value, name: &'static str) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => Ok(Value::CharArray(string_scalar_to_chars(&text))),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(Value::CharArray(string_scalar_to_chars(&array.data()[0])))
        }
        Value::StringArray(array) => {
            let values = array
                .data()
                .iter()
                .map(|text| Value::CharArray(string_scalar_to_chars(text)))
                .collect();
            crate::make_cell_with_shape(values, array.shape().to_vec()).map_err(|message| {
                crate::build_runtime_error(message)
                    .with_builtin(name)
                    .build()
            })
        }
        other => Ok(other),
    }
}
