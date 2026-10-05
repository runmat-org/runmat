use super::capabilities::{DEFAULT_DATETIME_FORMAT, DEFAULT_DATE_FORMAT};
use super::*;
use runmat_value::CellArray;

pub(super) async fn validate_cell_text_input(
    value: &Value,
    budget: Option<&mut crate::execution::CpuCooperationBudget>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    let Value::Cell(cells) = value else {
        return Ok(());
    };
    validate_cell_text_elements(cells, budget, runtime).await
}

async fn validate_cell_text_elements(
    cells: &CellArray,
    budget: Option<&mut crate::execution::CpuCooperationBudget>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    let mut budget = budget;
    for (index, element) in cells.data.iter().enumerate() {
        if let Some(budget) = budget.as_deref_mut() {
            budget
                .checkpoint(1)
                .await
                .map_err(|source| datetime_execution_error(runtime, source))?;
        }
        if !matches!(element, Value::CharArray(_))
            || !crate::builtins::common::validation::value_is_text_scalar(element)
        {
            return Err(datetime_argument_error(format!(
                "datetime: cell element {} must be a character vector",
                index + 1
            )));
        }
    }
    Ok(())
}

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
    parse_datetime_text_trimmed(text.trim())
}

fn parse_datetime_text_trimmed(trimmed: &str) -> Option<(NaiveDateTime, bool)> {
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
    parse_datetime_text_with_input_format_trimmed(text.trim(), input_format)
}

fn parse_datetime_text_with_input_format_trimmed(
    trimmed: &str,
    input_format: Option<&str>,
) -> Option<(NaiveDateTime, bool)> {
    if trimmed.is_empty() {
        return None;
    }
    let Some(input_format) = input_format else {
        return parse_datetime_text_trimmed(trimmed);
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
    value: &Value,
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
            let (naive, has_time) = parse_datetime_text_with_input_format(text, input_format)
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
                &Value::StringArray(
                    StringArray::new(texts, vec![array.rows, 1])
                        .map_err(|err| datetime_error(format!("datetime: {err}")))?,
                ),
                input_format,
            )
        }
        _ => Err(datetime_error(
            "datetime: text input must be a string scalar, string array, character array, or cell array of character vectors",
        )),
    }
}

pub(super) async fn parse_cell_text_input(
    value: &Value,
    input_format: Option<&str>,
    budget: Option<&mut crate::execution::CpuCooperationBudget>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<(Vec<f64>, Vec<usize>, String)> {
    let Value::Cell(cells) = value else {
        return Err(datetime_argument_error(
            "datetime: expected cell array of character vectors",
        ));
    };
    let mut budget = budget;
    let mut serials = Vec::with_capacity(cells.data.len());
    let mut has_time = false;
    for (index, element) in cells.data.iter().enumerate() {
        let Value::CharArray(chars) = element else {
            return Err(datetime_argument_error(format!(
                "datetime: cell element {} must be a character vector",
                index + 1
            )));
        };
        if !crate::builtins::common::validation::value_is_text_scalar(element) {
            return Err(datetime_argument_error(format!(
                "datetime: cell element {} must be a character vector",
                index + 1
            )));
        }
        let mut text = String::new();
        let mut first_non_whitespace = None;
        let mut last_non_whitespace_end = 0;
        for chunk in chars.data.chunks(1024) {
            if let Some(budget) = budget.as_deref_mut() {
                budget
                    .checkpoint(u64::try_from(chunk.len()).unwrap_or(u64::MAX))
                    .await
                    .map_err(|source| datetime_execution_error(runtime, source))?;
            }
            let additional = chunk.len().checked_mul(4).ok_or_else(|| {
                datetime_error("datetime: character vector allocation exceeds platform limits")
            })?;
            text.try_reserve(additional).map_err(|source| {
                let mut error = datetime_error("datetime: character vector allocation failed");
                error.source = Some(Box::new(source));
                error
            })?;
            for character in chunk {
                let start = text.len();
                text.push(*character);
                if !character.is_whitespace() {
                    first_non_whitespace.get_or_insert(start);
                    last_non_whitespace_end = text.len();
                }
            }
        }
        if chars.data.is_empty() {
            if let Some(budget) = budget.as_deref_mut() {
                budget
                    .checkpoint(1)
                    .await
                    .map_err(|source| datetime_execution_error(runtime, source))?;
            }
        }
        let trimmed = first_non_whitespace
            .map(|start| &text[start..last_non_whitespace_end])
            .unwrap_or("");
        let parsed = relative_datetime(trimmed)
            .map(|value| (value, trimmed.eq_ignore_ascii_case("now")))
            .or_else(|| parse_datetime_text_with_input_format_trimmed(trimmed, input_format));
        let (naive, parsed_has_time) = parsed.ok_or_else(|| {
            datetime_error(format!("datetime: unable to parse date/time text '{text}'"))
        })?;
        serials.push(datenum_from_naive(naive));
        has_time |= parsed_has_time;
    }
    Ok((
        serials,
        cells.shape.clone(),
        if has_time {
            DEFAULT_DATETIME_FORMAT
        } else {
            DEFAULT_DATE_FORMAT
        }
        .to_string(),
    ))
}

pub(super) fn relative_datetime(text: &str) -> Option<NaiveDateTime> {
    if ![3, 5, 8, 9].contains(&text.len()) {
        return None;
    }
    let now = Local::now().naive_local();
    if text.eq_ignore_ascii_case("now") {
        Some(now)
    } else if text.eq_ignore_ascii_case("today") {
        Some(midnight(now.date()))
    } else if text.eq_ignore_ascii_case("tomorrow") {
        Some(midnight(now.date() + Duration::days(1)))
    } else if text.eq_ignore_ascii_case("yesterday") {
        Some(midnight(now.date() - Duration::days(1)))
    } else {
        None
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
