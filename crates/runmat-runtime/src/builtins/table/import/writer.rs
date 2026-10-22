use super::*;
use crate::builtins::common::tensor;
use crate::RuntimeError;
use runmat_value::decode_utf16_code_units;

pub(in crate::builtins::table) fn table_delimited_text(
    object: &ObjectInstance,
    delimiter: &str,
    write_variable_names: bool,
    convert_row_times: bool,
) -> BuiltinResult<String> {
    let names = table_variable_names_from_object(object)?;
    let variables = table_variables_ref(object)?
        .ok_or_else(|| invalid_variable("table writer: variable storage is missing"))?;
    let row_times = if convert_row_times {
        Some(
            timetable_row_times_ref(object)?
                .ok_or_else(|| invalid_variable("writetimetable: RowTimes storage is missing"))?,
        )
    } else {
        None
    };
    let height = if let Some(times) = row_times {
        value_row_count(times)?
    } else if let Some(name) = names.first() {
        value_row_count(
            variables
                .fields
                .get(name)
                .ok_or_else(|| invalid_variable("writetable: variable storage is incomplete"))?,
        )?
    } else {
        0
    };
    for name in &names {
        let value = variables
            .fields
            .get(name)
            .ok_or_else(|| invalid_variable(format!("table writer: missing variable '{name}'")))?;
        if value_row_count(value)? != height {
            return Err(invalid_variable(
                "table writer: variables and row times must have equal height",
            ));
        }
    }
    let mut lines = Vec::new();
    if write_variable_names {
        let header = row_times
            .into_iter()
            .map(|_| "Time")
            .chain(names.iter().map(String::as_str))
            .map(|name| escape_delimited_field(name, delimiter))
            .collect::<Vec<_>>()
            .join(delimiter);
        lines.push(header);
    }
    for row in 0..height {
        let values = row_times
            .into_iter()
            .chain(names.iter().map(|name| &variables.fields[name]));
        lines.push(
            values
                .map(|value| {
                    row_value(value, row).and_then(|cell| {
                        cell_to_text(&cell).map(|text| escape_delimited_field(&text, delimiter))
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()?
                .join(delimiter),
        );
    }
    lines.push(String::new());
    Ok(lines.join("\n"))
}

pub(in crate::builtins::table) fn cell_to_text(value: &Value) -> BuiltinResult<String> {
    Ok(match value {
        Value::String(text) => text.to_string(),
        Value::CharArray(array) if array.rows == 1 => decode_utf16_code_units(array.data())
            .map_err(|error| table_utf16_error("table text value", error))?,
        Value::Num(value) => format_key_number(*value),
        Value::Int(value) => value.decimal_string(),
        Value::Bool(value) => {
            if *value {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0))
            .map(|value| value.decimal_string())
            .unwrap_or_else(|| format_key_number(tensor::tensor_value_f64(tensor, 0))),
        Value::StringArray(array) if array.data().len() == 1 => array.data()[0].to_string(),
        other => other.to_string(),
    })
}

pub(in crate::builtins::table) fn escape_delimited_field(text: &str, delimiter: &str) -> String {
    if text.contains(delimiter) || text.contains('"') || text.contains('\n') || text.contains('\r')
    {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

pub(in crate::builtins::table) fn char_rows(array: &CharArray) -> BuiltinResult<Vec<String>> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        rows.push(
            decode_utf16_code_units(&array.data()[start..start + array.cols])
                .map_err(|error| table_utf16_error("table character row", error))?,
        );
    }
    Ok(rows)
}

fn table_utf16_error(context: &str, error: runmat_value::Utf16DecodeError) -> RuntimeError {
    table_error_with_source(
        &TABLE_ERROR_INVALID_VARIABLE,
        format!("{context} is not valid UTF-16 text: {error}"),
        error,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntegerStorage, Tensor, Utf16CodeUnit};

    #[test]
    fn cell_to_text_preserves_exact_integer_scalar_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();

        assert_eq!(
            cell_to_text(&Value::Tensor(tensor)).unwrap(),
            "18446744073709551615"
        );
    }

    #[test]
    fn cell_to_text_preserves_utf16_decode_source() {
        let chars =
            CharArray::from_code_units(vec![Utf16CodeUnit::from_raw(0xd800)], 1, 1).unwrap();
        let error = cell_to_text(&Value::CharArray(chars)).unwrap_err();

        assert!(std::error::Error::source(&error)
            .is_some_and(|source| source.is::<runmat_value::Utf16DecodeError>()));
    }
}
