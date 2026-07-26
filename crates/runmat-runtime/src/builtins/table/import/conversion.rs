use super::*;
use crate::builtins::common::tensor;

pub(in crate::builtins::table) fn import_rows_to_cell(
    rows: Vec<Vec<ImportCell>>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let row_count = rows.len();
    let col_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut data = Vec::with_capacity(row_count.saturating_mul(col_count));
    for row in rows {
        for col in 0..col_count {
            let cell = row.get(col).cloned().unwrap_or(ImportCell::Empty);
            data.push(import_cell_value(cell, options));
        }
    }
    CellArray::new(data, row_count, col_count)
        .map(Value::Cell)
        .map_err(|err| invalid_variable(format!("readcell: {err}")))
}

pub(in crate::builtins::table) fn import_cell_value(
    cell: ImportCell,
    options: &ReadTableOptions,
) -> Value {
    match cell {
        ImportCell::Empty => Value::String(String::new()),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Value::String(String::new())
            } else if let Some(value) = parse_logical(token) {
                Value::Bool(value)
            } else if let Some(value) = parse_numeric(token) {
                Value::Num(value)
            } else {
                Value::String(token.to_string())
            }
        }
        ImportCell::Number(value) => Value::Num(value),
        ImportCell::Logical(value) => Value::Bool(value),
        ImportCell::DateTime(serial) => Value::Num(serial),
        ImportCell::Error(text) => Value::String(text),
    }
}

pub(in crate::builtins::table) fn import_rows_to_table(
    mut rows: Vec<Vec<ImportCell>>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut variable_names = options.variable_names.clone();
    let read_variable_names = options
        .read_variable_names
        .unwrap_or_else(|| variable_names.is_none() && should_read_variable_names(&rows, options));
    if variable_names.is_none() && read_variable_names && !rows.is_empty() {
        variable_names = Some(
            rows.remove(0)
                .into_iter()
                .map(|cell| cell.display_text())
                .collect(),
        );
    }

    let mut row_names = options.row_names.clone();
    if options.read_row_names && !rows.is_empty() {
        row_names = Some(
            rows.iter_mut()
                .map(|row| {
                    if row.is_empty() {
                        String::new()
                    } else {
                        row.remove(0).display_text()
                    }
                })
                .collect(),
        );
        if let Some(names) = variable_names.as_mut() {
            if !names.is_empty() {
                names.remove(0);
            }
        }
    }

    let column_count = import_column_count(&rows, &variable_names, options)?;
    let names = import_variable_names(variable_names, column_count, options);

    let mut columns = Vec::with_capacity(names.len());
    for col in 0..names.len() {
        let values = rows
            .iter()
            .map(|row| row.get(col).cloned().unwrap_or(ImportCell::Empty))
            .collect::<Vec<_>>();
        let requested_type = options
            .variable_types
            .as_ref()
            .and_then(|types| types.get(col))
            .copied();
        columns.push(import_column(values, options, requested_type)?);
    }
    table_from_columns_with_properties(names, columns, row_names)
}

pub(in crate::builtins::table) fn import_column_count(
    rows: &[Vec<ImportCell>],
    variable_names: &Option<Vec<String>>,
    options: &ReadTableOptions,
) -> BuiltinResult<usize> {
    let data_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let name_cols = variable_names.as_ref().map(Vec::len).unwrap_or(0);
    let type_cols = options.variable_types.as_ref().map(Vec::len).unwrap_or(0);
    if let Some(count) = options.num_variables {
        if name_cols > count {
            return Err(invalid_argument(
                "readtable: VariableNames length exceeds NumVariables",
            ));
        }
        if type_cols > count {
            return Err(invalid_argument(
                "readtable: VariableTypes length exceeds NumVariables",
            ));
        }
        return Ok(count);
    }
    Ok(data_cols.max(name_cols).max(type_cols))
}

pub(in crate::builtins::table) fn import_variable_names(
    variable_names: Option<Vec<String>>,
    column_count: usize,
    options: &ReadTableOptions,
) -> Vec<String> {
    match variable_names {
        Some(mut names) => {
            while names.len() < column_count {
                names.push(format!("Var{}", names.len() + 1));
            }
            names.truncate(column_count);
            if options.preserve_variable_names {
                make_unique_names(names)
            } else {
                make_unique_variable_names(names)
            }
        }
        None => generated_variable_names(column_count),
    }
}

pub(in crate::builtins::table) fn should_read_variable_names(
    rows: &[Vec<ImportCell>],
    options: &ReadTableOptions,
) -> bool {
    let Some(first) = rows.first() else {
        return false;
    };
    if first.is_empty() {
        return false;
    }
    let names = first
        .iter()
        .map(ImportCell::display_text)
        .map(|text| text.trim().to_string())
        .collect::<Vec<_>>();
    if names.iter().any(|name| name.is_empty()) {
        return false;
    }
    if first.iter().all(|cell| cell.is_likely_data_token(options)) {
        return false;
    }
    true
}

pub(in crate::builtins::table) fn import_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
    requested_type: Option<ImportVariableType>,
) -> BuiltinResult<Value> {
    match requested_type.unwrap_or(ImportVariableType::Auto) {
        ImportVariableType::Auto => infer_import_column(values, options),
        ImportVariableType::Numeric(dtype) => import_numeric_column(values, options, dtype),
        ImportVariableType::Integer(target) => import_integer_column(values, options, target),
        ImportVariableType::Logical => import_logical_column(values, options),
        ImportVariableType::Text(kind) => import_text_column(values, options, kind),
        ImportVariableType::CellStr => import_cellstr_column(values, options),
        ImportVariableType::Categorical => import_categorical_column(values, options),
        ImportVariableType::Datetime => import_datetime_column(values, options),
        ImportVariableType::Duration => import_duration_column(values, options),
    }
}

pub(in crate::builtins::table) fn import_integer_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
    target: crate::builtins::math::elementwise::integer_cast::IntegerTarget,
) -> BuiltinResult<Value> {
    let mut parsed = Vec::with_capacity(values.len());
    for value in &values {
        parsed.push(integer_from_import_cell(value, options, target)?);
    }
    Tensor::new_integer(target.storage(parsed), vec![values.len(), 1])
        .map(Value::Tensor)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

fn integer_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
    target: crate::builtins::math::elementwise::integer_cast::IntegerTarget,
) -> BuiltinResult<runmat_builtins::IntValue> {
    let numeric = |value| Ok(target.cast_scalar(value));
    match value {
        ImportCell::Empty => numeric(f64::NAN),
        ImportCell::Number(value) | ImportCell::DateTime(value) => numeric(*value),
        ImportCell::Logical(value) => numeric(f64::from(*value)),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                return numeric(f64::NAN);
            }
            if let Ok(value) = token.parse::<i128>() {
                return Ok(target.cast_i128(value));
            }
            parse_numeric(token)
                .map(|value| target.cast_scalar(value))
                .ok_or_else(|| {
                    invalid_variable(format!(
                        "readtable: cannot import '{token}' as {}",
                        target.class_name()
                    ))
                })
        }
        ImportCell::Error(text) => Err(invalid_variable(format!(
            "readtable: cannot import spreadsheet error '{text}' as {}",
            target.class_name()
        ))),
    }
}

pub(in crate::builtins::table) fn import_numeric_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    let mut numeric = Vec::with_capacity(values.len());
    for value in &values {
        let parsed = numeric_from_import_cell(value, options, dtype.class_name())?;
        numeric.push(cast_import_numeric(parsed, dtype));
    }
    let tensor = Tensor::new(numeric, vec![values.len(), 1])
        .map_err(|err| invalid_variable(format!("readtable: {err}")))?;
    Ok(Value::Tensor(tensor::coerce_tensor_dtype(tensor, dtype)))
}

pub(in crate::builtins::table) fn numeric_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
    context: &str,
) -> BuiltinResult<f64> {
    match value {
        ImportCell::Empty => Ok(f64::NAN),
        ImportCell::Number(value) => Ok(*value),
        ImportCell::Logical(value) => Ok(if *value { 1.0 } else { 0.0 }),
        ImportCell::DateTime(serial) => Ok(*serial),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Ok(f64::NAN)
            } else {
                parse_numeric(token).ok_or_else(|| {
                    invalid_variable(format!("readtable: cannot import '{token}' as {context}"))
                })
            }
        }
        ImportCell::Error(text) => Err(invalid_variable(format!(
            "readtable: cannot import spreadsheet error '{text}' as {context}"
        ))),
    }
}

pub(in crate::builtins::table) fn cast_import_numeric(value: f64, dtype: NumericDType) -> f64 {
    match dtype {
        NumericDType::F64 => value,
        NumericDType::F32 => (value as f32) as f64,
        NumericDType::I8 => value.round().clamp(i8::MIN as f64, i8::MAX as f64),
        NumericDType::I16 => value.round().clamp(i16::MIN as f64, i16::MAX as f64),
        NumericDType::I32 => value.round().clamp(i32::MIN as f64, i32::MAX as f64),
        NumericDType::I64 => value.round().clamp(i64::MIN as f64, i64::MAX as f64),
        NumericDType::U8 => {
            if value.is_finite() {
                value.round().clamp(0.0, u8::MAX as f64)
            } else {
                0.0
            }
        }
        NumericDType::U16 => {
            if value.is_finite() {
                value.round().clamp(0.0, u16::MAX as f64)
            } else {
                0.0
            }
        }
        NumericDType::U32 => {
            if value.is_finite() {
                value.round().clamp(0.0, u32::MAX as f64)
            } else {
                0.0
            }
        }
        NumericDType::U64 => {
            if value.is_finite() {
                value.round().clamp(0.0, u64::MAX as f64)
            } else {
                0.0
            }
        }
    }
}

pub(in crate::builtins::table) fn import_logical_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut logical = Vec::with_capacity(values.len());
    for value in &values {
        logical.push(logical_from_import_cell(value, options)?);
    }
    LogicalArray::new(logical, vec![values.len(), 1])
        .map(Value::LogicalArray)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

pub(in crate::builtins::table) fn logical_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> BuiltinResult<u8> {
    let flag = match value {
        ImportCell::Empty => false,
        ImportCell::Logical(value) => *value,
        ImportCell::Number(value) => *value != 0.0,
        ImportCell::DateTime(serial) => *serial != 0.0,
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                false
            } else if let Some(value) = parse_logical(token) {
                value
            } else if let Some(value) = parse_numeric(token) {
                value != 0.0
            } else {
                return Err(invalid_variable(format!(
                    "readtable: cannot import '{token}' as logical"
                )));
            }
        }
        ImportCell::Error(text) => {
            return Err(invalid_variable(format!(
                "readtable: cannot import spreadsheet error '{text}' as logical"
            )));
        }
    };
    Ok(u8::from(flag))
}

pub(in crate::builtins::table) fn import_text_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
    kind: TextImportType,
) -> BuiltinResult<Value> {
    let strings = import_text_values(values, options);
    match kind {
        TextImportType::String => StringArray::new(strings.clone(), vec![strings.len(), 1])
            .map(Value::StringArray)
            .map_err(|err| invalid_variable(format!("readtable: {err}"))),
        TextImportType::Char => import_char_column(strings),
    }
}

pub(in crate::builtins::table) fn import_text_values(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> Vec<String> {
    values
        .into_iter()
        .map(|value| {
            if value.is_missing(options) {
                String::new()
            } else {
                unquote(value.display_text().trim()).to_string()
            }
        })
        .collect()
}

pub(in crate::builtins::table) fn import_char_column(strings: Vec<String>) -> BuiltinResult<Value> {
    let rows = strings.len();
    let cols = strings
        .iter()
        .map(|text| text.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = vec![' '; rows * cols];
    for (row, text) in strings.iter().enumerate() {
        for (col, ch) in text.chars().enumerate() {
            data[row * cols + col] = ch;
        }
    }
    CharArray::new(data, rows, cols)
        .map(Value::CharArray)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

pub(in crate::builtins::table) fn import_cellstr_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let strings = import_text_values(values, options);
    let rows = strings.len();
    let cells = strings
        .into_iter()
        .map(|text| Value::CharArray(CharArray::new_row(&text)))
        .collect::<Vec<_>>();
    CellArray::new(cells, rows, 1)
        .map(Value::Cell)
        .map_err(|err| invalid_variable(format!("readtable: {err}")))
}

pub(in crate::builtins::table) fn import_categorical_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let strings = import_text_values(values, options);
    categorical_from_args(vec![Value::StringArray(
        StringArray::new(strings.clone(), vec![strings.len(), 1])
            .map_err(|err| invalid_variable(format!("readtable: {err}")))?,
    )])
}

pub(in crate::builtins::table) fn import_datetime_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    if matches!(options.datetime_type, DatetimeImportType::Text) {
        return import_text_column(values, options, options.text_type);
    }

    let mut serials = Vec::with_capacity(values.len());
    for value in &values {
        serials.push(datetime_serial_from_import_cell(value, options)?);
    }
    let tensor = Tensor::new(serials, vec![values.len(), 1])
        .map_err(|err| invalid_variable(format!("readtable: {err}")))?;
    if matches!(options.datetime_type, DatetimeImportType::ExcelDatenum) {
        Ok(Value::Tensor(tensor))
    } else {
        crate::builtins::datetime::datetime_object_from_serial_tensor(tensor, "yyyy-MM-dd HH:mm:ss")
    }
}

pub(in crate::builtins::table) fn datetime_serial_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> BuiltinResult<f64> {
    match value {
        ImportCell::Empty => Ok(f64::NAN),
        ImportCell::DateTime(serial) => Ok(*serial),
        ImportCell::Number(value) => Ok(*value),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Ok(f64::NAN)
            } else if let Some(serial) = parse_iso_datetime_to_datenum(token) {
                Ok(serial)
            } else if let Some(serial) = parse_numeric(token) {
                Ok(serial)
            } else {
                Err(invalid_variable(format!(
                    "readtable: cannot import '{token}' as datetime"
                )))
            }
        }
        ImportCell::Logical(_) => Err(invalid_variable(
            "readtable: cannot import logical value as datetime",
        )),
        ImportCell::Error(text) => Err(invalid_variable(format!(
            "readtable: cannot import spreadsheet error '{text}' as datetime"
        ))),
    }
}

pub(in crate::builtins::table) fn import_duration_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut days = Vec::with_capacity(values.len());
    for value in &values {
        days.push(duration_days_from_import_cell(value, options)?);
    }
    let tensor = Tensor::new(days, vec![values.len(), 1])
        .map_err(|err| invalid_variable(format!("readtable: {err}")))?;
    crate::builtins::duration::duration_object_from_days_tensor(
        tensor,
        crate::builtins::duration::DEFAULT_DURATION_FORMAT,
    )
}

pub(in crate::builtins::table) fn duration_days_from_import_cell(
    value: &ImportCell,
    options: &ReadTableOptions,
) -> BuiltinResult<f64> {
    match value {
        ImportCell::Empty => Ok(f64::NAN),
        ImportCell::Number(value) => Ok(*value),
        ImportCell::Logical(value) => Ok(if *value { 1.0 } else { 0.0 }),
        ImportCell::Text(text) => {
            let token = unquote(text.trim()).trim();
            if options.is_missing(token) {
                Ok(f64::NAN)
            } else {
                parse_duration_to_days(token).ok_or_else(|| {
                    invalid_variable(format!("readtable: cannot import '{token}' as duration"))
                })
            }
        }
        ImportCell::DateTime(_) => Err(invalid_variable(
            "readtable: cannot import datetime value as duration",
        )),
        ImportCell::Error(text) => Err(invalid_variable(format!(
            "readtable: cannot import spreadsheet error '{text}' as duration"
        ))),
    }
}

pub(in crate::builtins::table) fn infer_import_column(
    values: Vec<ImportCell>,
    options: &ReadTableOptions,
) -> BuiltinResult<Value> {
    let mut numeric = Vec::with_capacity(values.len());
    let mut all_numeric = true;
    for value in &values {
        match value {
            ImportCell::Empty => numeric.push(f64::NAN),
            ImportCell::Number(value) => numeric.push(*value),
            ImportCell::Text(text) => {
                let token = unquote(text.trim()).trim();
                if options.is_missing(token) {
                    numeric.push(f64::NAN);
                } else if let Some(value) = parse_numeric(token) {
                    numeric.push(value);
                } else {
                    all_numeric = false;
                    break;
                }
            }
            _ => {
                all_numeric = false;
                break;
            }
        }
    }
    if all_numeric {
        return Tensor::new(numeric, vec![values.len(), 1])
            .map(Value::Tensor)
            .map_err(|err| invalid_variable(format!("readtable: {err}")));
    }

    let mut logical = Vec::with_capacity(values.len());
    let mut all_logical = true;
    for value in &values {
        match value {
            ImportCell::Empty => logical.push(0),
            ImportCell::Logical(value) => logical.push(i32::from(*value) as u8),
            ImportCell::Text(text) => {
                let token = unquote(text.trim()).trim();
                if options.is_missing(token) {
                    logical.push(0);
                } else if let Some(value) = parse_logical(token) {
                    logical.push(i32::from(value) as u8);
                } else {
                    all_logical = false;
                    break;
                }
            }
            _ => {
                all_logical = false;
                break;
            }
        }
    }
    if all_logical {
        return LogicalArray::new(logical, vec![values.len(), 1])
            .map(Value::LogicalArray)
            .map_err(|err| invalid_variable(format!("readtable: {err}")));
    }

    if !matches!(options.datetime_type, DatetimeImportType::Text) {
        let mut serials = Vec::with_capacity(values.len());
        let mut all_datetime = true;
        for value in &values {
            match value {
                ImportCell::Empty => serials.push(f64::NAN),
                ImportCell::DateTime(serial) => serials.push(*serial),
                ImportCell::Text(text) => {
                    let token = unquote(text.trim()).trim();
                    if options.is_missing(token) {
                        serials.push(f64::NAN);
                    } else if let Some(serial) = parse_iso_datetime_to_datenum(token) {
                        serials.push(serial);
                    } else {
                        all_datetime = false;
                        break;
                    }
                }
                _ => {
                    all_datetime = false;
                    break;
                }
            }
        }
        if all_datetime {
            let tensor = Tensor::new(serials, vec![values.len(), 1])
                .map_err(|err| invalid_variable(format!("readtable: {err}")))?;
            if matches!(options.datetime_type, DatetimeImportType::ExcelDatenum) {
                return Ok(Value::Tensor(tensor));
            }
            return crate::builtins::datetime::datetime_object_from_serial_tensor(
                tensor,
                "yyyy-MM-dd HH:mm:ss",
            );
        }
    }

    import_text_column(values, options, options.text_type)
}

pub(in crate::builtins::table) fn parse_numeric(token: &str) -> Option<f64> {
    match token.to_ascii_lowercase().as_str() {
        "nan" => Some(f64::NAN),
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        _ => token.parse::<f64>().ok(),
    }
}

pub(in crate::builtins::table) fn parse_logical(token: &str) -> Option<bool> {
    match token.to_ascii_lowercase().as_str() {
        "true" | "t" | "yes" | "on" => Some(true),
        "false" | "f" | "no" | "off" => Some(false),
        _ => None,
    }
}

pub(in crate::builtins::table) fn parse_duration_to_days(token: &str) -> Option<f64> {
    parse_numeric(token).or_else(|| parse_clock_duration_to_days(token))
}

pub(in crate::builtins::table) fn parse_clock_duration_to_days(token: &str) -> Option<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (sign, body) = if let Some(rest) = trimmed.strip_prefix('-') {
        (-1.0, rest)
    } else if let Some(rest) = trimmed.strip_prefix('+') {
        (1.0, rest)
    } else {
        (1.0, trimmed)
    };
    let parts = body.split(':').collect::<Vec<_>>();
    let (hours, minutes, seconds) = match parts.as_slice() {
        [hours, minutes] => (
            hours.parse::<f64>().ok()?,
            minutes.parse::<f64>().ok()?,
            0.0,
        ),
        [hours, minutes, seconds] => (
            hours.parse::<f64>().ok()?,
            minutes.parse::<f64>().ok()?,
            seconds.parse::<f64>().ok()?,
        ),
        _ => return None,
    };
    if !hours.is_finite()
        || !minutes.is_finite()
        || !seconds.is_finite()
        || !(0.0..60.0).contains(&minutes)
        || !(0.0..60.0).contains(&seconds)
    {
        return None;
    }
    Some(sign * (hours * 3600.0 + minutes * 60.0 + seconds) / 86_400.0)
}

pub(in crate::builtins::table) fn parse_iso_datetime_to_datenum(token: &str) -> Option<f64> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return None;
    }
    for format in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y/%m/%d %H:%M:%S%.f",
        "%m/%d/%Y %H:%M:%S%.f",
    ] {
        if let Ok(value) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Some(crate::builtins::datetime::datenum_from_naive(value));
        }
    }
    for format in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"] {
        if let Ok(date) = NaiveDate::parse_from_str(trimmed, format) {
            return Some(crate::builtins::datetime::datenum_from_naive(
                date.and_time(NaiveTime::MIN),
            ));
        }
    }
    None
}

pub(in crate::builtins::table) fn unquote(token: &str) -> &str {
    if token.len() >= 2 {
        let bytes = token.as_bytes();
        if (bytes[0] == b'"' && bytes[token.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[token.len() - 1] == b'\'')
        {
            return &token[1..token.len() - 1];
        }
    }
    token
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::IntegerStorage;

    #[test]
    fn import_numeric_column_materializes_unsigned_typed_integer_storage() {
        let values = vec![
            ImportCell::Number(-5.0),
            ImportCell::Text("42.4".into()),
            ImportCell::Text("999".into()),
            ImportCell::Empty,
        ];
        let out = import_numeric_column(values, &ReadTableOptions::default(), NumericDType::U8)
            .expect("uint8 import");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor, got {out:?}");
        };
        assert_eq!(tensor.shape, vec![4, 1]);
        assert_eq!(tensor.dtype, NumericDType::U8);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U8(vec![0, 42, u8::MAX, 0]))
        );
    }

    #[test]
    fn import_numeric_column_materializes_signed_typed_integer_storage() {
        let values = vec![
            ImportCell::Text("-40000".into()),
            ImportCell::Number(-7.6),
            ImportCell::Logical(true),
            ImportCell::Number(40000.0),
        ];
        let out = import_numeric_column(values, &ReadTableOptions::default(), NumericDType::I16)
            .expect("int16 import");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor, got {out:?}");
        };
        assert_eq!(tensor.shape, vec![4, 1]);
        assert_eq!(tensor.dtype, NumericDType::I16);
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::I16(vec![i16::MIN, -8, 1, i16::MAX]))
        );
    }
}
