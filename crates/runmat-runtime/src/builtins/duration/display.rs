use super::*;

pub(super) fn format_seconds_field(seconds: f64) -> String {
    let whole = seconds.floor();
    let fractional = seconds - whole;
    if fractional.abs() <= 1e-9 {
        format!("{:02}", whole as i64)
    } else {
        let mut text = format!("{:06.3}", seconds);
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    }
}

pub(super) fn format_duration_value(days: f64, format: &str) -> BuiltinResult<String> {
    if days.is_nan() {
        return Ok("NaN".to_string());
    }
    if days == f64::INFINITY {
        return Ok("Inf".to_string());
    }
    if days == f64::NEG_INFINITY {
        return Ok("-Inf".to_string());
    }

    let total_seconds = days * SECONDS_PER_DAY;
    let sign = if total_seconds < 0.0 { "-" } else { "" };
    let total_seconds = total_seconds.abs();
    let total_hours = (total_seconds / 3600.0).floor();
    let total_minutes = (total_seconds / 60.0).floor();
    let hours = total_hours as i64;
    let minutes_component = ((total_seconds / 60.0).floor() as i64) % 60;
    let seconds_component =
        total_seconds - (hours as f64 * 3600.0) - (minutes_component as f64 * 60.0);

    let rendered = match format {
        "hh:mm:ss" => format!(
            "{sign}{hours:02}:{minutes_component:02}:{}",
            format_seconds_field(seconds_component)
        ),
        "hh:mm" => format!("{sign}{hours:02}:{minutes_component:02}"),
        "mm:ss" => format!(
            "{sign}{:02}:{}",
            total_minutes as i64,
            format_seconds_field(total_seconds - total_minutes * 60.0)
        ),
        "s" | "ss" => {
            let mut text = format!("{:.3}", total_seconds);
            while text.contains('.') && text.ends_with('0') {
                text.pop();
            }
            if text.ends_with('.') {
                text.pop();
            }
            format!("{sign}{text}")
        }
        other => {
            return Err(duration_error(format!(
                "duration: unsupported Format value '{other}'"
            )))
        }
    };

    Ok(rendered)
}

pub fn duration_string_array(value: &Value) -> BuiltinResult<Option<StringArray>> {
    let Value::Object(obj) = value else {
        return Ok(None);
    };
    if !obj.is_class(DURATION_CLASS) {
        return Ok(None);
    }
    let days = duration_tensor_from_duration_value(value)?;
    let format = format_for_object(obj);
    let day_values = tensor::tensor_values_f64_cow(&days);
    let mut strings = Vec::with_capacity(day_values.len());
    for value in day_values.iter() {
        strings.push(format_duration_value(*value, &format)?);
    }
    let shape = tensor::default_shape_for(&days.shape, day_values.len());
    let array = StringArray::new(strings, shape)
        .map_err(|err| duration_error(format!("duration: {err}")))?;
    Ok(Some(array))
}

pub fn duration_display_text(value: &Value) -> BuiltinResult<Option<String>> {
    let Some(array) = duration_string_array(value)? else {
        return Ok(None);
    };
    if array.data.len() == 1 {
        return Ok(Some(array.data[0].clone()));
    }

    let rows = array.rows;
    let cols = array.cols;
    let mut widths = vec![0usize; cols];
    for col in 0..cols {
        for row in 0..rows {
            let idx = row + col * rows;
            widths[col] = widths[col].max(array.data[idx].len());
        }
    }

    let mut lines = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut line = String::new();
        for col in 0..cols {
            if col > 0 {
                line.push_str("  ");
            }
            let idx = row + col * rows;
            let text = &array.data[idx];
            line.push_str(text);
            let padding = widths[col].saturating_sub(text.len());
            if padding > 0 {
                line.push_str(&" ".repeat(padding));
            }
        }
        lines.push(line);
    }

    Ok(Some(lines.join("\n")))
}

pub fn duration_summary(value: &Value) -> BuiltinResult<Option<String>> {
    let Value::Object(obj) = value else {
        return Ok(None);
    };
    if !obj.is_class(DURATION_CLASS) {
        return Ok(None);
    }
    let days = duration_tensor_from_duration_value(value)?;
    let len = days.len();
    if len == 1 {
        return duration_display_text(value);
    }
    let shape = tensor::default_shape_for(&days.shape, len);
    Ok(Some(format!(
        "[{} duration]",
        shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join("x")
    )))
}

pub fn duration_char_array(value: &Value) -> BuiltinResult<Option<CharArray>> {
    let Some(array) = duration_string_array(value)? else {
        return Ok(None);
    };
    let width = array.data.iter().map(String::len).max().unwrap_or(0);
    let rows = array.data.len();
    let mut data = vec![' '; rows * width];
    for (row, text) in array.data.iter().enumerate() {
        for (col, ch) in text.chars().enumerate() {
            data[row * width + col] = ch;
        }
    }
    let out = CharArray::new(data, rows, width)
        .map_err(|err| duration_error(format!("duration: {err}")))?;
    Ok(Some(out))
}
