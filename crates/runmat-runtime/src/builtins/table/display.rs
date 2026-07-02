use super::*;

pub fn table_display_text(value: &Value) -> BuiltinResult<String> {
    let object = match value {
        Value::Object(object) if is_tabular_class(object) => object,
        _ => return Err(invalid_argument("table display expects tabular object")),
    };
    let class_label = if object.is_class(TIMETABLE_CLASS) {
        TIMETABLE_CLASS
    } else {
        TABLE_CLASS
    };
    let names = table_variable_names_from_object(object)?;
    let variables = table_variables(object)?;
    let rows = table_height(object)?;
    let preview = rows.min(12);
    let mut widths = names.iter().map(|name| name.len()).collect::<Vec<_>>();
    let rendered_cols = names
        .iter()
        .enumerate()
        .map(|(col, name)| {
            let value = variables
                .fields
                .get(name)
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
            let cells = (0..preview)
                .map(|row| render_table_cell(&value, row))
                .collect::<Vec<_>>();
            for cell in &cells {
                widths[col] = widths[col].max(cell.len());
            }
            cells
        })
        .collect::<Vec<_>>();

    let mut lines = Vec::new();
    lines.push(format!("{rows}x{} {class_label}", names.len()));
    if names.is_empty() {
        return Ok(lines.join("\n"));
    }
    let header = names
        .iter()
        .enumerate()
        .map(|(idx, name)| format!("{name:<width$}", width = widths[idx]))
        .collect::<Vec<_>>()
        .join("  ");
    lines.push(header);
    for row in 0..preview {
        lines.push(
            rendered_cols
                .iter()
                .enumerate()
                .map(|(col, cells)| format!("{:<width$}", cells[row], width = widths[col]))
                .collect::<Vec<_>>()
                .join("  "),
        );
    }
    if preview < rows {
        lines.push(format!("... {} more rows", rows - preview));
    }
    Ok(lines.join("\n"))
}

pub fn table_summary_text(value: &Value) -> BuiltinResult<String> {
    let object = match value {
        Value::Object(object) if is_tabular_class(object) => object,
        _ => return Err(invalid_argument("table display expects tabular object")),
    };
    let class_label = if object.is_class(TIMETABLE_CLASS) {
        TIMETABLE_CLASS
    } else {
        TABLE_CLASS
    };
    Ok(format!(
        "{}x{} {}",
        table_height(object)?,
        table_width(object)?,
        class_label
    ))
}

pub(super) fn render_table_cell(value: &Value, row: usize) -> String {
    match value {
        Value::Tensor(tensor) => tensor
            .get2(row, 0)
            .map(format_table_number)
            .unwrap_or_default(),
        Value::StringArray(array) => array.data.get(row).cloned().unwrap_or_default(),
        Value::LogicalArray(array) => array
            .data
            .get(row)
            .map(|value| if *value != 0 { "true" } else { "false" }.to_string())
            .unwrap_or_default(),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::datetime_string_array(value)
                .ok()
                .flatten()
                .and_then(|array| array.data.get(row).cloned())
                .unwrap_or_else(|| value.to_string())
        }
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => {
            categorical_label_at(obj, row).unwrap_or_else(|| value.to_string())
        }
        other => other.to_string(),
    }
}

pub(crate) fn categorical_label_at(object: &ObjectInstance, row: usize) -> Option<String> {
    let code = match object.properties.get("Codes")? {
        Value::Tensor(tensor) => tensor.get2(row, 0).ok()?,
        _ => return None,
    };
    if !code.is_finite() || code < 1.0 {
        return Some("<undefined>".to_string());
    }
    let idx = code.round() as usize - 1;
    match object.properties.get("Categories")? {
        Value::StringArray(array) => array.data.get(idx).cloned(),
        _ => None,
    }
}

pub(super) fn format_table_number(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        trim_float(format!("{value:.6}"))
    }
}

pub(super) fn format_key_number(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        value.to_string()
    } else {
        trim_float(format!("{value:.17}"))
    }
}

pub(super) fn trim_float(mut text: String) -> String {
    if let Some(dot) = text.find('.') {
        let mut end = text.len();
        while end > dot + 1 && text.as_bytes()[end - 1] == b'0' {
            end -= 1;
        }
        if end == dot + 1 {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}
