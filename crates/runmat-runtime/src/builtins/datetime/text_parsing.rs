use super::capabilities::{DEFAULT_DATETIME_FORMAT, DEFAULT_DATE_FORMAT};
use super::*;
use runmat_value::CellArray;
use std::borrow::Cow;

mod datestr_format;
pub(super) use self::datestr_format::parse_datestr_format;

pub(super) fn decode_datetime_row(units: &[runmat_value::Utf16CodeUnit]) -> BuiltinResult<String> {
    runmat_value::decode_utf16_code_units(units).map_err(datetime_utf16_error)
}

pub(super) fn utf16_chunk_end(units: &[runmat_value::Utf16CodeUnit], start: usize) -> usize {
    let mut end = start.saturating_add(1024).min(units.len());
    if end < units.len() && (0xD800..=0xDBFF).contains(&units[end - 1].raw()) {
        end += 1;
    }
    end
}

pub(super) fn encode_utf16_chunk(
    units: &[runmat_value::Utf16CodeUnit],
    source_offset: usize,
    target: &mut [u8],
) -> Result<usize, runmat_value::Utf16DecodeError> {
    let mut encoded_len = 0usize;
    let mut unit_index = source_offset;
    for decoded in char::decode_utf16(units.iter().map(|unit| unit.raw())) {
        let character = decoded.map_err(|error| runmat_value::Utf16DecodeError {
            index: unit_index,
            unpaired_surrogate: error.unpaired_surrogate(),
        })?;
        let mut bytes = [0u8; 4];
        let encoded = character.encode_utf8(&mut bytes).as_bytes();
        target[encoded_len..encoded_len + encoded.len()].copy_from_slice(encoded);
        encoded_len += encoded.len();
        unit_index += character.len_utf16();
    }
    Ok(encoded_len)
}

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
    for (index, element) in cells.data().iter().enumerate() {
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

    for (pattern, has_time) in [
        ("%Y-%m-%d %H:%M:%S", true),
        ("%Y-%m-%d %H:%M", true),
        ("%Y/%m/%d %H:%M:%S%.f", true),
        ("%Y/%m/%d %H:%M:%S", true),
        ("%Y/%m/%d %H:%M", true),
        ("%Y-%m-%d", false),
        ("%d-%b-%Y %H:%M:%S", true),
        ("%d-%b-%Y %H:%M", true),
        ("%d-%b-%Y", false),
        ("%m/%d/%Y %H:%M:%S", true),
        ("%m/%d/%Y %H:%M", true),
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
    crate::builtins::common::validation::parse_datetime_with_input_format(trimmed, input_format)
}

pub(super) fn parse_text_input_with_runtime(
    value: &Value,
    input_format: Option<&str>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<(Vec<f64>, Vec<usize>, String)> {
    match value {
        Value::String(text) => {
            let text = nonmissing_text(text, "date/time input")?;
            if let Some(relative) = relative_datetime(text.trim(), runtime)? {
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
            let (naive, has_time) = parse_calendar_text(text, input_format, runtime)?
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
            let mut serials = Vec::with_capacity(array.data().len());
            let mut has_time = false;
            for text in array.data() {
                let text = nonmissing_text(text, "date/time input")?;
                let parsed = relative_datetime(text.trim(), runtime)?
                    .map(|value| (value, text.trim().eq_ignore_ascii_case("now")))
                    .or(parse_calendar_text(text, input_format, runtime)?);
                let (naive, parsed_has_time) = parsed.ok_or_else(|| {
                    datetime_error(format!("datetime: unable to parse date/time text '{text}'"))
                })?;
                serials.push(datenum_from_naive(naive));
                has_time |= parsed_has_time;
            }
            Ok((
                serials,
                tensor::default_shape_for(&array.shape, array.data().len()),
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
                texts.push(decode_datetime_row(&array.data()[start..end])?.trim_end().to_string());
            }
            parse_text_input_with_runtime(
                &Value::StringArray(
                    StringArray::new(texts, vec![array.rows, 1])
                        .map_err(|err| datetime_error(format!("datetime: {err}")))?,
                ),
                input_format,
                runtime,
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
    let mut serials = Vec::with_capacity(cells.data().len());
    let mut has_time = false;
    for (index, element) in cells.data().iter().enumerate() {
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
        if let Some(budget) = budget.as_deref_mut() {
            budget
                .checkpoint(1)
                .await
                .map_err(|source| datetime_execution_error(runtime, source))?;
        }
        let mut text = String::new();
        let capacity = chars.data().len().checked_mul(4).ok_or_else(|| {
            datetime_error("datetime: character vector allocation exceeds platform limits")
        })?;
        text.try_reserve(capacity).map_err(|source| {
            let mut error = datetime_error("datetime: character vector allocation failed");
            error.source = Some(Box::new(source));
            error
        })?;
        let mut unit_index = 0usize;
        let mut last_checkpoint = 0usize;
        let mut first_non_whitespace = None;
        let mut last_non_whitespace_end = 0usize;
        for decoded in char::decode_utf16(chars.data().iter().map(|unit| unit.raw())) {
            if unit_index.saturating_sub(last_checkpoint) >= 1024 {
                if let Some(budget) = budget.as_deref_mut() {
                    budget
                        .checkpoint(u64::try_from(unit_index - last_checkpoint).unwrap_or(u64::MAX))
                        .await
                        .map_err(|source| datetime_execution_error(runtime, source))?;
                }
                last_checkpoint = unit_index;
            }
            let character = decoded.map_err(|source| {
                datetime_utf16_error(runmat_value::Utf16DecodeError {
                    index: unit_index,
                    unpaired_surrogate: source.unpaired_surrogate(),
                })
            })?;
            let start = text.len();
            text.push(character);
            if !character.is_whitespace() {
                first_non_whitespace.get_or_insert(start);
                last_non_whitespace_end = text.len();
            }
            unit_index += character.len_utf16();
        }
        if let Some(budget) = budget.as_deref_mut() {
            budget
                .checkpoint(
                    u64::try_from(unit_index - last_checkpoint)
                        .unwrap_or(u64::MAX)
                        .max(1),
                )
                .await
                .map_err(|source| datetime_execution_error(runtime, source))?;
        }
        let trimmed = first_non_whitespace
            .map(|start| &text[start..last_non_whitespace_end])
            .unwrap_or("");
        let parsed = relative_datetime(trimmed, runtime)?
            .map(|value| (value, trimmed.eq_ignore_ascii_case("now")))
            .or(parse_calendar_text(trimmed, input_format, runtime)?);
        let (naive, parsed_has_time) = parsed.ok_or_else(|| {
            datetime_error(format!("datetime: unable to parse date/time text '{text}'"))
        })?;
        serials.push(datenum_from_naive(naive));
        has_time |= parsed_has_time;
    }
    Ok((
        serials,
        cells.shape().to_vec(),
        if has_time {
            DEFAULT_DATETIME_FORMAT
        } else {
            DEFAULT_DATE_FORMAT
        }
        .to_string(),
    ))
}

pub(super) fn relative_datetime(
    text: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Option<NaiveDateTime>> {
    if !["now", "today", "tomorrow", "yesterday"]
        .iter()
        .any(|name| text.eq_ignore_ascii_case(name))
    {
        return Ok(None);
    }
    let now = current_naive_local(runtime)?;
    Ok(if text.eq_ignore_ascii_case("now") {
        Some(now)
    } else if text.eq_ignore_ascii_case("today") {
        Some(midnight(now.date()))
    } else if text.eq_ignore_ascii_case("tomorrow") {
        Some(midnight(
            now.date()
                .checked_add_signed(Duration::days(1))
                .ok_or_else(|| datetime_error("tomorrow is outside the calendar range"))?,
        ))
    } else if text.eq_ignore_ascii_case("yesterday") {
        Some(midnight(
            now.date()
                .checked_sub_signed(Duration::days(1))
                .ok_or_else(|| datetime_error("yesterday is outside the calendar range"))?,
        ))
    } else {
        None
    })
}

fn parse_calendar_text(
    text: &str,
    input_format: Option<&str>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Option<(NaiveDateTime, bool)>> {
    let instant = input_format
        .is_none()
        .then(|| DateTime::parse_from_rfc3339(text.trim()))
        .and_then(Result::ok);
    if let Some(instant) = instant {
        let runtime = runtime
            .ok_or_else(|| datetime_error("timezone conversion requires a runtime context"))?;
        let instant = crate::context::RuntimeCalendarInstant::from_naive_utc(instant.naive_utc());
        return runtime
            .service_ports()
            .timing()?
            .local_datetime_at_instant(instant)
            .map(|local| Some((local, true)))
            .ok_or_else(|| {
                datetime_error("timezone conversion is unavailable under the runtime clock policy")
            });
    }
    Ok(parse_datetime_text_with_input_format(text, input_format))
}

/// Parse one borrowed datenum input without manufacturing a text array,
/// serial vector, or format string. Character rows are decoded into an
/// admitted temporary UTF-8 backing by the owning calendar route.
pub(super) fn parse_datenum_text_borrowed(
    text: &str,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<f64> {
    let trimmed = text.trim();
    let parsed = relative_datetime(trimmed, runtime)?
        .or(parse_calendar_text(trimmed, None, runtime)?.map(|(date, _)| date))
        .ok_or_else(|| {
            datetime_error(format!("datetime: unable to parse date/time text '{text}'"))
        })?;
    Ok(datenum_from_naive(parsed))
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

pub(super) fn map_legacy_component_text(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    input_format: Option<&str>,
    label: &str,
    extractor: impl Fn(&NaiveDateTime) -> f64,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    let (texts, shape): (Vec<Cow<'_, str>>, Vec<usize>) = match &value {
        Value::String(text) => (
            vec![Cow::Borrowed(nonmissing_text(text, "legacy date input")?)],
            vec![1, 1],
        ),
        Value::StringArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data().len());
            (
                array
                    .data()
                    .iter()
                    .map(|text| nonmissing_text(text, "legacy date input").map(Cow::Borrowed))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                shape,
            )
        }
        Value::CharArray(array) => {
            let texts = (0..array.rows)
                .map(|row| {
                    decode_datetime_row(&array.data()[row * array.cols..(row + 1) * array.cols])
                        .map(|text| Cow::Owned(text.trim_end().to_string()))
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            (texts, vec![array.rows, 1])
        }
        _ => {
            return Err(datetime_error(format!(
                "{label}: expected legacy date text"
            )))
        }
    };
    let legacy_format = input_format.map(legacy_date_format_to_strftime);
    let mut components = Vec::with_capacity(texts.len());
    for text in texts {
        let parsed = parse_legacy_component_text_borrowed(
            runtime,
            &text,
            legacy_format.as_deref(),
            label,
            false,
        )?;
        components.push(extractor(&parsed));
    }
    Ok((components, shape))
}

/// Parse one borrowed legacy component element. The owning route supplies any
/// decoded character row and the optional already-converted format backing.
pub(super) fn parse_legacy_component_text_borrowed(
    runtime: Option<&crate::context::RuntimeContext>,
    text: &str,
    format: Option<&str>,
    label: &str,
    date_only_format: bool,
) -> BuiltinResult<NaiveDateTime> {
    let parsed = if let Some(format) = format {
        if date_only_format {
            NaiveDate::parse_from_str(text.trim(), format)
                .ok()
                .map(|date| date.and_hms_opt(0, 0, 0).unwrap())
        } else {
            NaiveDateTime::parse_from_str(text.trim(), format)
                .ok()
                .or_else(|| {
                    NaiveDate::parse_from_str(text.trim(), format)
                        .ok()
                        .map(|date| date.and_hms_opt(0, 0, 0).unwrap())
                })
        }
    } else {
        relative_datetime(text.trim(), runtime)?
            .or(parse_calendar_text(text, None, runtime)?.map(|(date, _)| date))
    }
    .ok_or_else(|| {
        datetime_error(format!(
            "{label}: unable to parse legacy date text '{text}'"
        ))
    })?;
    Ok(parsed)
}

pub(super) fn parse_legacy_day_text(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    input_format: Option<&str>,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    let (texts, shape): (Vec<Cow<'_, str>>, Vec<usize>) = match &value {
        Value::String(text) => (
            vec![Cow::Borrowed(nonmissing_text(text, "legacy date input")?)],
            vec![1, 1],
        ),
        Value::StringArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data().len());
            (
                array
                    .data()
                    .iter()
                    .map(|text| nonmissing_text(text, "legacy date input").map(Cow::Borrowed))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                shape,
            )
        }
        Value::CharArray(array) => {
            let mut texts = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let text =
                    decode_datetime_row(&array.data()[row * array.cols..(row + 1) * array.cols])?;
                texts.push(Cow::Owned(text.trim_end().to_string()));
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
            relative_datetime(text.trim(), runtime)?
                .or(parse_calendar_text(&text, None, runtime)?.map(|(value, _)| value))
        }
        .ok_or_else(|| datetime_error(format!("day: unable to parse legacy date text '{text}'")))?;
        serials.push(datenum_from_naive(parsed));
    }
    Ok((serials, shape))
}
