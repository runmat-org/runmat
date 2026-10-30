//! Pure datetime-domain conversion for timer absolute firing-time forms.
//!
//! The timer owns scheduling; this module owns MATLAB date representation.
//! It does not read a clock or invoke another builtin.

use super::*;
use chrono::{Datelike, NaiveTime};

pub(crate) fn timer_firing_arguments(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
    reference_local: NaiveDateTime,
) -> BuiltinResult<Tensor> {
    match args.len() {
        1 => {
            let value = args.into_iter().next().unwrap();
            if is_datetime_object(&value) {
                if let Value::Object(object) = &value {
                    if object
                        .properties()
                        .get("TimeZone")
                        .is_some_and(|zone| !matches!(zone, Value::String(text) if text.as_str().is_some_and(str::is_empty)))
                    {
                        return Err(datetime_error(
                            "startat: timezone-bearing datetime requires an explicit instant model",
                        ));
                    }
                }
                return serials_from_datetime_value(&value);
            }
            if matches!(
                &value,
                Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
            ) {
                return timer_text_serials(&value, reference_local);
            }
            if matches!(&value, Value::Tensor(tensor) if tensor.shape().len() == 2 && matches!(tensor.shape()[1], 3 | 6))
            {
                let datevec = tensor_from_datevec_like(value, "startat")?;
                let (rows, _) = datevec_matrix_dimensions(&datevec);
                let values = datevec.materialize_f64();
                let mut serials = Vec::with_capacity(rows);
                for row in 0..rows {
                    let component = |column: usize| values[column * rows + row];
                    serials.push(serial_from_normalized_components(
                        component(0),
                        clamp_startat_month(component(1)),
                        component(2),
                        component(3),
                        component(4),
                        component(5),
                    )?);
                }
                return Tensor::new(serials, vec![rows, 1])
                    .map_err(|error| datetime_error(format!("startat: {error}")));
            }
            serial_tensor_from_value(value, "startat")
        }
        3 | 6 => {
            let month = component_tensor(args[1].clone(), "month")?;
            let shape = month.shape().to_vec();
            let values = month
                .materialize_f64()
                .into_iter()
                .map(clamp_startat_month)
                .collect();
            let month = Value::Tensor(
                Tensor::new(values, shape)
                    .map_err(|error| datetime_error(format!("startat: {error}")))?,
            );
            let source = args.into_function_arguments();
            let args = crate::execution::RootedValues::transform_arguments(
                &source,
                source.len(),
                None,
                runtime,
                |index| {
                    Ok(if index == 1 {
                        month.clone()
                    } else {
                        source.get(index).expect("timer component index").clone()
                    })
                },
            )?;
            let datetime =
                build_from_components(runtime, crate::call::BuiltinArguments::from(args), None)?;
            serials_from_datetime_value(&datetime)
        }
        _ => Err(datetime_error(
            "startat: expected one absolute firing time or Y/M/D[/H/MI/S] components",
        )),
    }
}

#[cfg(test)]
fn timer_firing_serials(args: Vec<Value>, reference_local: NaiveDateTime) -> BuiltinResult<Tensor> {
    let args = crate::execution::RootedValues::new(args)
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    timer_firing_arguments(None, args.into(), reference_local)
}

fn clamp_startat_month(month: f64) -> f64 {
    if month.is_finite() && month < 1.0 {
        1.0
    } else {
        month
    }
}

fn timer_text_serials(value: &Value, reference: NaiveDateTime) -> BuiltinResult<Tensor> {
    let (texts, shape): (Vec<String>, Vec<usize>) = match value {
        Value::String(text) => (
            vec![text
                .as_str()
                .ok_or_else(|| datetime_error("startat: firing time must not be missing"))?
                .to_owned()],
            vec![1, 1],
        ),
        Value::StringArray(array) => (
            array
                .data()
                .iter()
                .map(|text| {
                    text.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| datetime_error("startat: firing time must not be missing"))
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            array.shape.as_slice().to_vec(),
        ),
        Value::CharArray(array) => (
            (0..array.rows)
                .map(|row| {
                    super::text_parsing::decode_datetime_row(
                        &array.data()[row * array.cols..(row + 1) * array.cols],
                    )
                    .map(|text| text.trim_end().to_string())
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            vec![array.rows, 1],
        ),
        _ => unreachable!("text admission precedes timer text decoding"),
    };
    let serials = texts
        .iter()
        .map(|text| parse_timer_text(text, reference).map(datenum_from_naive))
        .collect::<BuiltinResult<Vec<_>>>()?;
    Tensor::new(serials, shape).map_err(|error| datetime_error(format!("startat: {error}")))
}

fn parse_timer_text(text: &str, reference: NaiveDateTime) -> BuiltinResult<NaiveDateTime> {
    let text = text.trim();
    if ["now", "today", "tomorrow", "yesterday"]
        .iter()
        .any(|word| text.eq_ignore_ascii_case(word))
        || chrono::DateTime::parse_from_rfc3339(text).is_ok()
    {
        return Err(datetime_error(
            "startat: relative words or timezone text require a separate instant authority",
        ));
    }
    // Slash-date syntax is decoded before the generic datetime parser: the
    // latter accepts short `%Y` fields literally, bypassing MATLAB's moving
    // two-digit-year pivot for the documented mm/dd/yy representation.
    let (date_part, time_part) = text
        .split_once(' ')
        .map_or((text, None), |(date, time)| (date, Some(time.trim())));
    if date_part.contains('/') {
        let date = parse_timer_slash_date(date_part, reference)?;
        let time = match time_part {
            None => NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
            Some(time) => parse_timer_time(time).ok_or_else(|| {
                datetime_error(format!("startat: invalid firing-time text '{text}'"))
            })?,
        };
        return Ok(date.and_time(time));
    }
    if let Some((naive, _)) = parse_datetime_text(text) {
        return Ok(naive);
    }
    if let Some(time) = parse_timer_time(text) {
        return Ok(reference.date().and_time(time));
    }
    Err(datetime_error(format!(
        "startat: unable to parse firing-time text '{text}'"
    )))
}

fn parse_timer_slash_date(text: &str, reference: NaiveDateTime) -> BuiltinResult<NaiveDate> {
    let parts: Vec<&str> = text.split('/').collect();
    let invalid = || datetime_error(format!("startat: invalid firing-date text '{text}'"));
    let (year, month, day) = match parts.as_slice() {
        [month, day] => (
            reference.year(),
            month.parse::<u32>().map_err(|_| invalid())?,
            day.parse::<u32>().map_err(|_| invalid())?,
        ),
        [first, second, third] if first.len() == 4 => (
            first.parse::<i32>().map_err(|_| invalid())?,
            second.parse::<u32>().map_err(|_| invalid())?,
            third.parse::<u32>().map_err(|_| invalid())?,
        ),
        [month, day, year] => {
            let month = month.parse::<u32>().map_err(|_| invalid())?;
            let day = day.parse::<u32>().map_err(|_| invalid())?;
            let year = if year.len() == 2 {
                let short = year.parse::<i32>().map_err(|_| invalid())?;
                let floor = reference.year() - 50;
                let mut expanded = floor.div_euclid(100) * 100 + short;
                if expanded < floor {
                    expanded += 100;
                }
                expanded
            } else if year.len() == 4 {
                year.parse::<i32>().map_err(|_| invalid())?
            } else {
                return Err(invalid());
            };
            (year, month, day)
        }
        _ => return Err(invalid()),
    };
    NaiveDate::from_ymd_opt(year, month, day).ok_or_else(invalid)
}

fn parse_timer_time(text: &str) -> Option<NaiveTime> {
    [
        "%H:%M:%S%.f",
        "%H:%M:%S",
        "%I:%M:%S %p",
        "%H:%M",
        "%I:%M %p",
    ]
    .iter()
    .find_map(|format| NaiveTime::parse_from_str(text, format).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2026, 10, 5)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
    }

    #[test]
    fn calendar_components_and_datevec_share_normalized_calendar_authority() {
        let components = timer_firing_serials(
            vec![Value::Num(2023.0), Value::Num(13.0), Value::Num(1.0)],
            reference(),
        )
        .expect("component rollover");
        let datevec = timer_firing_serials(
            vec![Value::Tensor(
                Tensor::new(vec![2023.0, 13.0, 1.0], vec![1, 3]).unwrap(),
            )],
            reference(),
        )
        .expect("datevec rollover");
        assert_eq!(components.materialize_f64(), datevec.materialize_f64());
        let expected = chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        assert_eq!(
            components.materialize_f64(),
            vec![datenum_from_naive(expected)]
        );
        let clamped = timer_firing_serials(
            vec![Value::Num(2024.0), Value::Num(0.0), Value::Num(5.0)],
            reference(),
        )
        .expect("startat clamps month below one");
        let january = NaiveDate::from_ymd_opt(2024, 1, 5)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        assert_eq!(clamped.materialize_f64(), vec![datenum_from_naive(january)]);
    }

    #[test]
    fn relative_words_and_explicit_timezone_are_not_reinterpreted_as_local_naive() {
        for text in ["now", "tomorrow", "2024-01-01T00:00:00+02:00"] {
            assert!(timer_firing_serials(vec![Value::String(text.into())], reference()).is_err());
        }
    }

    #[test]
    fn text_forms_use_injected_local_date_and_fifty_year_pivot() {
        let current = reference();
        let strings = [
            ("05-Oct-2026 14:30:05", (2026, 10, 5, 14, 30, 5)),
            ("05-Oct-2026", (2026, 10, 5, 0, 0, 0)),
            ("10/06", (2026, 10, 6, 0, 0, 0)),
            ("10/06/75", (2075, 10, 6, 0, 0, 0)),
            ("10/06/76", (1976, 10, 6, 0, 0, 0)),
            ("10/06/0075", (75, 10, 6, 0, 0, 0)),
            ("10/06/2075", (2075, 10, 6, 0, 0, 0)),
            ("10/06/2026", (2026, 10, 6, 0, 0, 0)),
            ("14:30:05", (2026, 10, 5, 14, 30, 5)),
            ("02:30:05 PM", (2026, 10, 5, 14, 30, 5)),
            ("14:30", (2026, 10, 5, 14, 30, 0)),
            ("02:30 PM", (2026, 10, 5, 14, 30, 0)),
        ];
        for (text, (year, month, day, hour, minute, second)) in strings {
            let parsed = timer_firing_serials(vec![Value::String(text.into())], current)
                .unwrap_or_else(|error| panic!("{text}: {error}"));
            let expected = NaiveDate::from_ymd_opt(year, month, day)
                .unwrap()
                .and_hms_opt(hour, minute, second)
                .unwrap();
            assert_eq!(parsed.materialize_f64(), vec![datenum_from_naive(expected)]);
        }
        for invalid in ["10/06/7", "10/06/123", "10/06/20760", "10/32/26"] {
            assert!(timer_firing_serials(vec![Value::String(invalid.into())], current).is_err());
        }
    }
}
