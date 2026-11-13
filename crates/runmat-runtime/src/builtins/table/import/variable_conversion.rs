//! Apply the shared per-variable import recipe before the existing column
//! conversion owner selects its output class. No second CSV parser is involved.

use super::*;
use crate::builtins::io::tabular::import_options::{EmptyFieldRule, VariableImportOptions};

pub(super) fn prepare_variable_cells(
    mut values: Vec<ImportCell>,
    options: &ReadTableOptions,
    column: usize,
    requested_type: Option<ImportVariableType>,
) -> BuiltinResult<Vec<ImportCell>> {
    let Some(variable) = options.variable_options.get(column) else {
        return Ok(values);
    };
    for value in &mut values {
        let current = std::mem::replace(value, ImportCell::Empty);
        *value = prepare_cell(current, options, variable, requested_type)?;
    }
    Ok(values)
}

fn prepare_cell(
    value: ImportCell,
    options: &ReadTableOptions,
    variable: &VariableImportOptions,
    requested_type: Option<ImportVariableType>,
) -> BuiltinResult<ImportCell> {
    let missing = match &value {
        ImportCell::Empty | ImportCell::Missing => true,
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            options.is_missing(token) || variable.is_missing(token)
        }
        _ => false,
    };
    if missing {
        if matches!(value, ImportCell::Empty) && variable.empty_field_rule == EmptyFieldRule::Error
        {
            return Err(invalid_variable(format!(
                "readtable: empty field in variable '{}'",
                variable.name
            )));
        }
        return match &variable.fill_value {
            Some(Value::Int(value))
                if matches!(requested_type, Some(ImportVariableType::Integer(_))) =>
            {
                Ok(ImportCell::Text(value.to_i128().to_string()))
            }
            Some(fill) => fill_cell(fill),
            None => Ok(if matches!(value, ImportCell::Missing) {
                ImportCell::Missing
            } else {
                ImportCell::Empty
            }),
        };
    }
    let ImportCell::Text(text) = value else {
        return Ok(value);
    };
    let token = unquote(text.trim()).trim();
    if matches!(requested_type, Some(ImportVariableType::Datetime)) {
        if let Some(format) = variable.input_format.as_deref() {
            return parse_datetime_with_format(token, format)
                .map(ImportCell::DateTime)
                .ok_or_else(|| {
                    invalid_variable(format!(
                        "readtable: '{}' does not match InputFormat for variable '{}'",
                        token, variable.name
                    ))
                });
        }
    }
    if matches!(
        requested_type,
        None | Some(
            ImportVariableType::Auto
                | ImportVariableType::Numeric(_)
                | ImportVariableType::Integer(_)
        )
    ) && (variable.decimal_separator.is_some() || variable.thousands_separator.is_some())
    {
        if matches!(requested_type, Some(ImportVariableType::Integer(_))) {
            return Ok(ImportCell::Text(
                crate::builtins::io::tabular::import_options::normalize_numeric_token(
                    token,
                    variable.decimal_separator.unwrap_or('.'),
                    variable.thousands_separator.flatten(),
                )
                .into_owned(),
            ));
        }
        if let Some(number) = variable.numeric(token) {
            return Ok(ImportCell::Number(number));
        }
    }
    Ok(ImportCell::Text(text))
}

fn fill_cell(value: &Value) -> BuiltinResult<ImportCell> {
    match value {
        Value::Num(value) => Ok(ImportCell::Number(*value)),
        Value::Int(value) => Ok(ImportCell::Number(value.to_f64())),
        Value::Bool(value) => Ok(ImportCell::Logical(*value)),
        Value::String(value) => Ok(match value.as_str() {
            Some(text) => ImportCell::Text(text.to_owned()),
            None => ImportCell::Missing,
        }),
        Value::StringArray(array) if array.data().len() == 1 => {
            fill_cell(&Value::String(array.data()[0].clone()))
        }
        Value::CharArray(value) if value.rows == 1 => {
            runmat_value::decode_utf16_code_units(value.data())
                .map(ImportCell::Text)
                .map_err(|error| {
                    table_error_with_source(
                        &TABLE_ERROR_INVALID_VARIABLE,
                        format!("readtable: FillValue is not valid UTF-16 text: {error}"),
                        error,
                    )
                })
        }
        _ => Err(invalid_variable(
            "readtable: unsupported variable FillValue class",
        )),
    }
}

fn parse_datetime_with_format(token: &str, format: &str) -> Option<f64> {
    let (datetime, _) =
        crate::builtins::common::validation::parse_datetime_with_input_format(token, format)?;
    Some(crate::builtins::datetime::datenum_from_naive(datetime))
}
