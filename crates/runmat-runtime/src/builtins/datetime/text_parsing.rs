use super::capabilities::{DEFAULT_DATETIME_FORMAT, DEFAULT_DATE_FORMAT};
use super::*;

pub(super) fn format_serial(serial: f64, format: &str) -> BuiltinResult<String> {
    if serial.is_nan() {
        return Ok("NaT".to_string());
    }
    if serial.is_infinite() {
        return Ok(if serial.is_sign_positive() {
            "Inf"
        } else {
            "-Inf"
        }
        .to_string());
    }
    let naive = naive_from_datenum(serial)?;
    let chrono_format = format_token_to_strftime(format);
    Ok(naive.format(&chrono_format).to_string())
}

pub(super) fn parse_datetime_text(text: &str) -> Option<(NaiveDateTime, bool)> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(value) = DateTime::parse_from_rfc3339(trimmed) {
        return Some((value.with_timezone(&Local).naive_local(), true));
    }

    for (pattern, has_time) in [
        ("%Y-%m-%d %H:%M:%S", true),
        ("%Y/%m/%d %H:%M:%S%.f", true),
        ("%Y/%m/%d %H:%M:%S", true),
        ("%Y-%m-%d", false),
        ("%d-%b-%Y %H:%M:%S", true),
        ("%d-%b-%Y", false),
        ("%m/%d/%Y %H:%M:%S", true),
        ("%m/%d/%Y", false),
    ] {
        if has_time {
            if let Ok(value) = NaiveDateTime::parse_from_str(trimmed, pattern) {
                return Some((value, true));
            }
        } else if let Ok(value) = NaiveDate::parse_from_str(trimmed, pattern) {
            return Some((value.and_hms_opt(0, 0, 0).unwrap(), false));
        }
    }

    None
}

pub(super) fn parse_datetime_text_with_input_format(
    text: &str,
    input_format: Option<&str>,
) -> Option<(NaiveDateTime, bool)> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let Some(input_format) = input_format else {
        return parse_datetime_text(trimmed);
    };
    let chrono_format = format_token_to_strftime(input_format);
    if let Ok(value) = NaiveDateTime::parse_from_str(trimmed, &chrono_format) {
        return Some((value, true));
    }
    if let Ok(value) = NaiveDate::parse_from_str(trimmed, &chrono_format) {
        return Some((value.and_hms_opt(0, 0, 0).unwrap(), false));
    }
    None
}

pub(super) fn parse_text_input(
    value: Value,
    input_format: Option<&str>,
) -> BuiltinResult<(Vec<f64>, Vec<usize>, String)> {
    match value {
        Value::String(text) => {
            if let Some(relative) = relative_datetime(text.trim()) {
                let has_time = text.trim().eq_ignore_ascii_case("now");
                return Ok((
                    vec![datenum_from_naive(relative)],
                    vec![1, 1],
                    if has_time {
                        DEFAULT_DATETIME_FORMAT
                    } else {
                        DEFAULT_DATE_FORMAT
                    }
                    .to_string(),
                ));
            }
            let (naive, has_time) = parse_datetime_text_with_input_format(&text, input_format)
                .ok_or_else(|| {
                    datetime_error(format!("datetime: unable to parse date/time text '{text}'"))
                })?;
            Ok((
                vec![datenum_from_naive(naive)],
                vec![1, 1],
                if has_time {
                    DEFAULT_DATETIME_FORMAT.to_string()
                } else {
                    DEFAULT_DATE_FORMAT.to_string()
                },
            ))
        }
        Value::StringArray(array) => {
            let mut serials = Vec::with_capacity(array.data.len());
            let mut has_time = false;
            for text in &array.data {
                let parsed = relative_datetime(text.trim())
                    .map(|value| (value, text.trim().eq_ignore_ascii_case("now")))
                    .or_else(|| parse_datetime_text_with_input_format(text, input_format));
                let (naive, parsed_has_time) = parsed.ok_or_else(|| {
                    datetime_error(format!("datetime: unable to parse date/time text '{text}'"))
                })?;
                serials.push(datenum_from_naive(naive));
                has_time |= parsed_has_time;
            }
            Ok((
                serials,
                tensor::default_shape_for(&array.shape, array.data.len()),
                if has_time {
                    DEFAULT_DATETIME_FORMAT.to_string()
                } else {
                    DEFAULT_DATE_FORMAT.to_string()
                },
            ))
        }
        Value::CharArray(array) => {
            let mut texts = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let start = row * array.cols;
                let end = start + array.cols;
                texts.push(
                    array.data[start..end]
                        .iter()
                        .collect::<String>()
                        .trim_end()
                        .to_string(),
                );
            }
            parse_text_input(
                Value::StringArray(
                    StringArray::new(texts, vec![array.rows, 1])
                        .map_err(|err| datetime_error(format!("datetime: {err}")))?,
                ),
                input_format,
            )
        }
        _ => Err(datetime_error(
            "datetime: text input must be a string scalar, string array, or character array",
        )),
    }
}

pub(super) fn relative_datetime(text: &str) -> Option<NaiveDateTime> {
    let now = Local::now().naive_local();
    match text.to_ascii_lowercase().as_str() {
        "now" => Some(now),
        "today" => Some(midnight(now.date())),
        "tomorrow" => Some(midnight(now.date() + Duration::days(1))),
        "yesterday" => Some(midnight(now.date() - Duration::days(1))),
        _ => None,
    }
}

pub(super) fn legacy_date_format_to_strftime(format: &str) -> String {
    let mut out = format.to_string();
    for (source, target) in [
        (".fff", "%.3f"),
        ("yyyy", "%Y"),
        ("mmmm", "%B"),
        ("mmm", "%b"),
        ("HH", "%H"),
        ("hh", "%H"),
        ("MM", "%M"),
        ("ss", "%S"),
        ("mm", "%m"),
        ("dd", "%d"),
    ] {
        out = out.replace(source, target);
    }
    out
}

pub(super) fn parse_legacy_component_text(
    value: Value,
    input_format: Option<&str>,
    label: &str,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    let (texts, shape) = match value {
        Value::String(text) => (vec![text], vec![1, 1]),
        Value::StringArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data.len());
            (array.data, shape)
        }
        Value::CharArray(array) => {
            let texts = (0..array.rows)
                .map(|row| {
                    array.data[row * array.cols..(row + 1) * array.cols]
                        .iter()
                        .collect::<String>()
                        .trim_end()
                        .to_string()
                })
                .collect();
            (texts, vec![array.rows, 1])
        }
        _ => {
            return Err(datetime_error(format!(
                "{label}: expected legacy date text"
            )))
        }
    };
    let legacy_format = input_format.map(legacy_date_format_to_strftime);
    let mut serials = Vec::with_capacity(texts.len());
    for text in texts {
        let parsed = if let Some(format) = legacy_format.as_deref() {
            NaiveDateTime::parse_from_str(text.trim(), format)
                .ok()
                .or_else(|| {
                    NaiveDate::parse_from_str(text.trim(), format)
                        .ok()
                        .map(|date| date.and_hms_opt(0, 0, 0).unwrap())
                })
        } else {
            parse_datetime_text(text.trim()).map(|(value, _)| value)
        }
        .ok_or_else(|| {
            datetime_error(format!(
                "{label}: unable to parse legacy date text '{text}'"
            ))
        })?;
        serials.push(datenum_from_naive(parsed));
    }
    Ok((serials, shape))
}

pub(super) fn parse_legacy_day_text(
    value: Value,
    input_format: Option<&str>,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    let (texts, shape) = match value {
        Value::String(text) => (vec![text], vec![1, 1]),
        Value::StringArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data.len());
            (array.data, shape)
        }
        Value::CharArray(array) => {
            let mut texts = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                texts.push(
                    array.data[row * array.cols..(row + 1) * array.cols]
                        .iter()
                        .collect::<String>()
                        .trim_end()
                        .to_string(),
                );
            }
            (texts, vec![array.rows, 1])
        }
        _ => return Err(datetime_error("day: expected legacy date text")),
    };
    let legacy_format = input_format.map(legacy_date_format_to_strftime);
    let mut serials = Vec::with_capacity(texts.len());
    for text in texts {
        let parsed = if let Some(format) = legacy_format.as_deref() {
            NaiveDate::parse_from_str(text.trim(), format)
                .ok()
                .map(|date| date.and_hms_opt(0, 0, 0).unwrap())
        } else {
            parse_datetime_text(text.trim()).map(|(value, _)| value)
        }
        .ok_or_else(|| datetime_error(format!("day: unable to parse legacy date text '{text}'")))?;
        serials.push(datenum_from_naive(parsed));
    }
    Ok((serials, shape))
}
