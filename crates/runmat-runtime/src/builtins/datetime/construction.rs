use super::capabilities::{DEFAULT_DATETIME_FORMAT, DEFAULT_DATE_FORMAT};
use super::*;

pub(super) fn round_component(value: f64, label: &str, min: i64, max: i64) -> BuiltinResult<i64> {
    if !value.is_finite() {
        return Err(datetime_error(format!(
            "datetime: {label} values must be finite"
        )));
    }
    let rounded = value.round();
    if (rounded - value).abs() > 1e-9 {
        return Err(datetime_error(format!(
            "datetime: {label} values must be integers"
        )));
    }
    let integer = rounded as i64;
    if integer < min || integer > max {
        return Err(datetime_error(format!(
            "datetime: {label} values must be in the range [{min}, {max}]"
        )));
    }
    Ok(integer)
}

pub(super) fn naive_from_components(
    year: f64,
    month: f64,
    day: f64,
    hour: f64,
    minute: f64,
    second: f64,
) -> BuiltinResult<NaiveDateTime> {
    let year = round_component(year, "year", -262_000, 262_000)? as i32;
    let month = round_component(month, "month", 1, 12)? as u32;
    let day = round_component(day, "day", 1, 31)? as u32;
    let hour = round_component(hour, "hour", 0, 23)? as u32;
    let minute = round_component(minute, "minute", 0, 59)? as u32;
    if !second.is_finite() {
        return Err(datetime_error("datetime: second values must be finite"));
    }
    if !(0.0..60.0).contains(&second) {
        return Err(datetime_error(
            "datetime: second values must be in the range [0, 60)",
        ));
    }

    let base_date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| datetime_error("datetime: invalid calendar date"))?;
    let whole_second = second.floor();
    let mut nanos = ((second - whole_second) * 1_000_000_000.0).round() as u32;
    let mut secs = whole_second as u32;
    if nanos == 1_000_000_000 {
        secs += 1;
        nanos = 0;
    }
    let time = base_date
        .and_hms_nano_opt(hour, minute, secs, nanos)
        .ok_or_else(|| datetime_error("datetime: invalid time components"))?;
    Ok(time)
}

pub(super) fn broadcast_component_data(
    arrays: &[Tensor],
    labels: &[&str],
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let mut target_shape = vec![1, 1];
    let mut target_len = 1usize;

    for array in arrays {
        let len = tensor::tensor_element_len(array);
        if len > 1 {
            let shape = tensor::default_shape_for(&array.shape, len);
            if target_len == 1 {
                target_len = len;
                target_shape = shape;
            } else if len != target_len || shape != target_shape {
                return Err(datetime_error(
                    "datetime: non-scalar component inputs must have matching sizes",
                ));
            }
        }
    }

    let mut broadcasted = Vec::with_capacity(arrays.len());
    for (idx, array) in arrays.iter().enumerate() {
        let len = tensor::tensor_element_len(array);
        if len == 1 {
            broadcasted.push(vec![tensor::tensor_value_f64(array, 0); target_len]);
        } else if len == target_len {
            broadcasted.push(tensor::tensor_values_f64(array));
        } else {
            return Err(datetime_error(format!(
                "datetime: {} input size does not match the other components",
                labels[idx]
            )));
        }
    }

    Ok((broadcasted, target_shape))
}

pub(super) fn component_tensor(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor_from_numeric(value, context)?;
    validate_authoritative_integer_storage(&tensor, context)?;
    let tensor = tensor::integer_tensor_to_f64(tensor)
        .map_err(|err| datetime_error(format!("datetime: {err}")))?;
    let shape = tensor::default_shape_for(&tensor.shape, tensor::tensor_element_len(&tensor));
    let values = tensor::tensor_into_values_f64(tensor);
    Tensor::new(values, shape).map_err(|err| datetime_error(format!("datetime: {err}")))
}

pub(super) fn build_from_components(
    args: Vec<Value>,
    format: Option<String>,
) -> BuiltinResult<Value> {
    let labels = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
    ];
    let input_count = args.len();
    let mut arrays = Vec::with_capacity(args.len());
    for (idx, arg) in args.into_iter().enumerate() {
        arrays.push(component_tensor(arg, labels[idx])?);
    }
    while arrays.len() < 7 {
        arrays.push(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    }

    let (broadcasted, shape) = broadcast_component_data(&arrays, &labels)?;
    let len = broadcasted[0].len();
    let mut serials = Vec::with_capacity(len);
    for idx in 0..len {
        if input_count == 7 && broadcasted[5][idx].is_finite() && broadcasted[5][idx].fract() != 0.0
        {
            return Err(datetime_error(
                "datetime: second must be integral when a millisecond component is supplied",
            ));
        }
        let second = broadcasted[5][idx] + broadcasted[6][idx] / 1_000.0;
        let serial = serial_from_normalized_components(
            broadcasted[0][idx],
            broadcasted[1][idx],
            broadcasted[2][idx],
            broadcasted[3][idx],
            broadcasted[4][idx],
            second,
        )?;
        serials.push(serial);
    }

    let default_format = if let Some(format) = format {
        format
    } else if input_count > 3 {
        DEFAULT_DATETIME_FORMAT.to_string()
    } else {
        DEFAULT_DATE_FORMAT.to_string()
    };
    datetime_object_from_serials(serials, shape, default_format)
}

pub(super) fn serial_from_normalized_components(
    year: f64,
    month: f64,
    day: f64,
    hour: f64,
    minute: f64,
    second: f64,
) -> BuiltinResult<f64> {
    let components = [year, month, day, hour, minute, second];
    if components.iter().any(|value| value.is_nan()) {
        return Ok(f64::NAN);
    }
    if let Some(infinite) = components.iter().find(|value| value.is_infinite()) {
        return Ok(*infinite);
    }
    let year = round_integral_component(year, "year")?;
    let month = round_integral_component(month, "month")?;
    let day = round_integral_component(day, "day")?;
    let hour = round_integral_component(hour, "hour")?;
    let minute = round_integral_component(minute, "minute")?;
    if !second.is_finite() {
        return Ok(second);
    }
    if year < i64::from(i32::MIN) || year > i64::from(i32::MAX) {
        return Err(datetime_error(
            "datetime: year is outside the supported range",
        ));
    }
    let month_offset = month.checked_sub(1).ok_or_else(|| {
        datetime_error("datetime: calendar components are outside the supported range")
    })?;
    let month_index = year
        .checked_mul(12)
        .and_then(|value| value.checked_add(month_offset))
        .ok_or_else(|| {
            datetime_error("datetime: calendar components are outside the supported range")
        })?;
    let normalized_year = month_index.div_euclid(12);
    let normalized_month = month_index.rem_euclid(12) as u32 + 1;
    let base = NaiveDate::from_ymd_opt(
        i32::try_from(normalized_year)
            .map_err(|_| datetime_error("datetime: year is outside the supported range"))?,
        normalized_month,
        1,
    )
    .ok_or_else(|| datetime_error("datetime: calendar components are outside the supported range"))?
    .and_hms_opt(0, 0, 0)
    .unwrap();
    let whole_seconds = second.trunc();
    let fractional_seconds = second - whole_seconds;
    if whole_seconds < i64::MIN as f64 || whole_seconds > i64::MAX as f64 {
        return Err(datetime_error(
            "datetime: second is outside the supported range",
        ));
    }
    let nanos = (fractional_seconds * 1_000_000_000.0).round();
    if nanos < i64::MIN as f64 || nanos > i64::MAX as f64 {
        return Err(datetime_error(
            "datetime: fractional second is outside the supported range",
        ));
    }
    let day_offset = day.checked_sub(1).ok_or_else(|| {
        datetime_error("datetime: calendar components are outside the supported range")
    })?;
    let normalized = base
        .checked_add_signed(Duration::days(day_offset))
        .and_then(|value| value.checked_add_signed(Duration::hours(hour)))
        .and_then(|value| value.checked_add_signed(Duration::minutes(minute)))
        .and_then(|value| value.checked_add_signed(Duration::seconds(whole_seconds as i64)))
        .and_then(|value| value.checked_add_signed(Duration::nanoseconds(nanos as i64)))
        .ok_or_else(|| {
            datetime_error("datetime: normalized components are outside the supported range")
        })?;
    Ok(datenum_from_naive(normalized))
}

pub(super) fn round_integral_component(value: f64, label: &str) -> BuiltinResult<i64> {
    let rounded = value.round();
    if (rounded - value).abs() > 1e-9 || rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return Err(datetime_error(format!(
            "datetime: {label} values must be representable integers"
        )));
    }
    Ok(rounded as i64)
}

pub(super) fn build_from_date_vectors(
    value: Value,
    format: Option<String>,
) -> BuiltinResult<Value> {
    let tensor = tensor_from_numeric(value, "date vector")?;
    validate_authoritative_integer_storage(&tensor, "date vector")?;
    let shape = tensor.shape.clone();
    if shape.len() != 2 || !matches!(shape[1], 3 | 6) {
        return Err(datetime_error(
            "datetime: a numeric one-input date vector must be an m-by-3 or m-by-6 matrix",
        ));
    }
    let rows = shape[0];
    let cols = shape[1];
    let values = tensor::tensor_values_f64_cow(&tensor);
    let mut serials = Vec::with_capacity(rows);
    for row in 0..rows {
        let at = |col: usize, default: f64| {
            if col < cols {
                values[row + col * rows]
            } else {
                default
            }
        };
        serials.push(serial_from_normalized_components(
            at(0, 0.0),
            at(1, 1.0),
            at(2, 1.0),
            at(3, 0.0),
            at(4, 0.0),
            at(5, 0.0),
        )?);
    }
    datetime_object_from_serials(
        serials,
        vec![rows, 1],
        format.unwrap_or_else(|| {
            if cols == 6 {
                DEFAULT_DATETIME_FORMAT
            } else {
                DEFAULT_DATE_FORMAT
            }
            .to_string()
        }),
    )
}

pub(super) fn numeric_value_to_datetime(
    value: Value,
    format: Option<String>,
) -> BuiltinResult<Value> {
    let serials = serial_tensor_from_value(value, "datetime")?;
    datetime_object_from_serial_tensor(
        serials,
        format.unwrap_or_else(|| DEFAULT_DATETIME_FORMAT.to_string()),
    )
}
