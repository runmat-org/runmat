use super::*;
use crate::builtins::common::tensor;

pub(in crate::builtins::table) async fn write_tabular_file(
    value: Value,
    rest: Vec<Value>,
    convert_row_times: bool,
) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(invalid_argument("writetable: filename is required"));
    }
    let path = resolve_path(&rest[0])?;
    let delimiter = parse_named_option(&rest[1..], "Delimiter")
        .map(|value| scalar_text(value, "Delimiter"))
        .transpose()?
        .unwrap_or_else(|| ",".to_string());
    let write_variable_names = parse_bool_option(
        &strip_known_text_option(&rest[1..], "Delimiter")?,
        "WriteVariableNames",
        true,
        "writetable",
    )?;
    let object = into_table_object(value, "writetable")?;
    let text = table_delimited_text(&object, &delimiter, write_variable_names, convert_row_times)?;
    let bytes = text.into_bytes();
    runmat_filesystem::write_async(path, &bytes)
        .await
        .map_err(|err| {
            table_error_with_source(&TABLE_ERROR_IO, "writetable: file write failed", err)
        })?;
    Ok(Value::Num(bytes.len() as f64))
}

pub(in crate::builtins::table) fn strip_known_text_option(
    args: &[Value],
    name: &str,
) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "writetable: name-value options must be provided in pairs",
            ));
        }
        let option_name = scalar_text(&args[idx], "option name")?;
        if !option_name.eq_ignore_ascii_case(name) {
            out.push(args[idx].clone());
            out.push(args[idx + 1].clone());
        }
        idx += 2;
    }
    Ok(out)
}

pub(in crate::builtins::table) fn table_delimited_text(
    object: &ObjectInstance,
    delimiter: &str,
    write_variable_names: bool,
    convert_row_times: bool,
) -> BuiltinResult<String> {
    let mut names = table_variable_names_from_object(object)?;
    let variables = table_variables(object)?;
    let mut columns = Vec::with_capacity(names.len() + usize::from(convert_row_times));
    if convert_row_times {
        if let Some(row_times) = timetable_row_times(object)? {
            names.insert(0, "Time".to_string());
            columns.push(row_times);
        }
    }
    for name in &table_variable_names_from_object(object)? {
        columns.push(
            variables.fields.get(name).cloned().ok_or_else(|| {
                invalid_variable(format!("writetable: missing variable '{name}'"))
            })?,
        );
    }
    let height = table_height(object)?;
    let mut lines = Vec::new();
    if write_variable_names {
        lines.push(
            names
                .iter()
                .map(|name| escape_delimited_field(name, delimiter))
                .collect::<Vec<_>>()
                .join(delimiter),
        );
    }
    for row in 0..height {
        lines.push(
            columns
                .iter()
                .map(|value| {
                    row_value(value, row)
                        .map(|cell| escape_delimited_field(&cell_to_text(&cell), delimiter))
                })
                .collect::<BuiltinResult<Vec<_>>>()?
                .join(delimiter),
        );
    }
    lines.push(String::new());
    Ok(lines.join("\n"))
}

pub(in crate::builtins::table) fn cell_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::CharArray(array) if array.rows == 1 => array.data.iter().collect(),
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
        Value::StringArray(array) if array.data.len() == 1 => array.data[0].clone(),
        other => other.to_string(),
    }
}

pub(in crate::builtins::table) fn escape_delimited_field(text: &str, delimiter: &str) -> String {
    if text.contains(delimiter) || text.contains('"') || text.contains('\n') || text.contains('\r')
    {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

pub(in crate::builtins::table) fn char_rows(array: &CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        rows.push(array.data[start..start + array.cols].iter().collect());
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntegerStorage, Tensor};

    #[test]
    fn cell_to_text_preserves_exact_integer_scalar_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();

        assert_eq!(cell_to_text(&Value::Tensor(tensor)), "18446744073709551615");
    }
}
