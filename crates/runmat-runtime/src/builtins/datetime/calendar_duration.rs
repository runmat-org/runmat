use super::capabilities::{
    BUILTIN_NAME, CALENDAR_DAYS_FIELD, CALENDAR_DURATION_CLASS, CALENDAR_MONTHS_FIELD,
    DATETIME_CLASS, DEFAULT_DATETIME_FORMAT, SECONDS_PER_DAY,
};
use super::*;

pub fn is_datetime_object(value: &Value) -> bool {
    matches!(value, Value::Object(obj) if obj.is_class(DATETIME_CLASS))
}

pub fn is_calendar_duration_object(value: &Value) -> bool {
    matches!(value, Value::Object(obj) if obj.is_class(CALENDAR_DURATION_CLASS))
}

pub(super) fn calendar_duration_tensor_for_object(
    obj: &ObjectInstance,
    field: &str,
) -> BuiltinResult<Tensor> {
    match obj.properties.get(field) {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        Some(Value::Num(value)) => Tensor::new(vec![*value], vec![1, 1])
            .map_err(|err| datetime_error(format!("calendarDuration: {err}"))),
        Some(other) => Err(datetime_error(format!(
            "calendarDuration: invalid internal {field} storage {other:?}"
        ))),
        None => Err(datetime_error(format!(
            "calendarDuration: missing internal {field} storage"
        ))),
    }
}

pub(crate) fn calendar_duration_tensors_from_value(
    value: &Value,
) -> BuiltinResult<(Tensor, Tensor)> {
    match value {
        Value::Object(obj) if obj.is_class(CALENDAR_DURATION_CLASS) => Ok((
            calendar_duration_tensor_for_object(obj, CALENDAR_MONTHS_FIELD)?,
            calendar_duration_tensor_for_object(obj, CALENDAR_DAYS_FIELD)?,
        )),
        _ => Err(datetime_error(
            "calendarDuration: expected a calendarDuration value",
        )),
    }
}

pub(crate) fn calendar_duration_object_from_tensors(
    months: Tensor,
    days: Tensor,
) -> BuiltinResult<Value> {
    ensure_calendar_duration_class_registered();
    let mut object = ObjectInstance::new(CALENDAR_DURATION_CLASS.to_string());
    object
        .properties
        .insert(CALENDAR_MONTHS_FIELD.to_string(), Value::Tensor(months));
    object
        .properties
        .insert(CALENDAR_DAYS_FIELD.to_string(), Value::Tensor(days));
    Ok(Value::Object(object))
}

pub(super) fn calendar_duration_object_from_components(
    months: Vec<f64>,
    days: Vec<f64>,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    let month_tensor = Tensor::new(months, shape.clone())
        .map_err(|err| datetime_error(format!("calendarDuration: {err}")))?;
    let day_tensor = Tensor::new(days, shape)
        .map_err(|err| datetime_error(format!("calendarDuration: {err}")))?;
    calendar_duration_object_from_tensors(month_tensor, day_tensor)
}

pub(super) fn calendar_duration_unit_value(
    value: Value,
    unit_name: &str,
    months_per_unit: f64,
    days_per_unit: f64,
) -> BuiltinResult<Value> {
    if is_calendar_duration_object(&value) {
        let (months, days) = calendar_duration_tensors_from_value(&value)?;
        let (month_data, day_data, shape) =
            tensor::binary_numeric_tensors(&months, &days, unit_name, BUILTIN_NAME)?;
        let data = month_data
            .iter()
            .zip(day_data.iter())
            .map(|(months, days)| {
                if months_per_unit != 0.0 {
                    months / months_per_unit + days / 30.436875 / months_per_unit
                } else {
                    days / days_per_unit
                }
            })
            .collect::<Vec<_>>();
        return tensor_or_scalar(data, shape);
    }

    let numeric = component_tensor(value, unit_name)?;
    let values = tensor::tensor_values_f64_cow(&numeric);
    let shape = tensor::default_shape_for(&numeric.shape, values.len());
    let mut months = Vec::with_capacity(values.len());
    let mut days = Vec::with_capacity(values.len());
    for value in values.iter().copied() {
        if !value.is_finite() {
            return Err(datetime_error(format!(
                "{unit_name}: values must be finite"
            )));
        }
        let month_value = value * months_per_unit;
        let day_value = value * days_per_unit;
        if !month_value.is_finite() || !day_value.is_finite() {
            return Err(datetime_error(format!(
                "{unit_name}: resulting calendar duration is outside supported range"
            )));
        }
        months.push(month_value);
        days.push(day_value);
    }
    calendar_duration_object_from_components(months, days, shape)
}

pub(super) fn add_months_clamped(
    value: NaiveDateTime,
    month_delta: i64,
) -> BuiltinResult<NaiveDateTime> {
    let current_month = i64::from(value.year())
        .checked_mul(12)
        .and_then(|base| base.checked_add(i64::from(value.month() - 1)))
        .ok_or_else(|| datetime_error("calendarDuration: result date is out of range"))?;
    let zero_based = current_month
        .checked_add(month_delta)
        .ok_or_else(|| datetime_error("calendarDuration: result date is out of range"))?;
    let year_i64 = zero_based.div_euclid(12);
    let year = i32::try_from(year_i64)
        .map_err(|_| datetime_error("calendarDuration: result date is out of range"))?;
    let month = zero_based.rem_euclid(12) as u32 + 1;
    let day = value.day().min(days_in_month(year, month)?);
    NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|date| {
            date.and_hms_nano_opt(
                value.hour(),
                value.minute(),
                value.second(),
                value.nanosecond(),
            )
        })
        .ok_or_else(|| datetime_error("calendarDuration: result date is out of range"))
}

pub(super) fn add_fractional_days(value: NaiveDateTime, days: f64) -> BuiltinResult<NaiveDateTime> {
    if !days.is_finite() {
        return Err(datetime_error(
            "calendarDuration: day components must be finite",
        ));
    }
    let nanos = (days * SECONDS_PER_DAY * 1_000_000_000.0).round();
    if !nanos.is_finite() || nanos < i64::MIN as f64 || nanos > i64::MAX as f64 {
        return Err(datetime_error(
            "calendarDuration: day component is outside supported range",
        ));
    }
    Ok(value + Duration::nanoseconds(nanos as i64))
}

pub(super) fn apply_calendar_duration_to_serials(
    serials: &Tensor,
    months: &Tensor,
    days: &Tensor,
    sign: f64,
    context: &str,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    let (serial_data, month_data, day_data, shape) =
        broadcast_three_numeric_tensors(serials, months, days, context)?;
    let mut out = Vec::with_capacity(serial_data.len());
    for ((serial, months), days) in serial_data
        .iter()
        .zip(month_data.iter())
        .zip(day_data.iter())
    {
        if !months.is_finite() {
            return Err(datetime_error(format!(
                "{context}: month components must be finite"
            )));
        }
        let signed_months = months * sign;
        let rounded_months = signed_months.round();
        if (rounded_months - signed_months).abs() > 1e-9 {
            return Err(datetime_error(format!(
                "{context}: calendar month components must be integers for datetime arithmetic"
            )));
        }
        if rounded_months < i64::MIN as f64 || rounded_months > i64::MAX as f64 {
            return Err(datetime_error(format!(
                "{context}: calendar month component is outside supported range"
            )));
        }
        let shifted = add_months_clamped(naive_from_datenum(*serial)?, rounded_months as i64)?;
        out.push(datenum_from_naive(add_fractional_days(
            shifted,
            days * sign,
        )?));
    }
    Ok((out, shape))
}

pub(crate) fn serials_from_datetime_value(value: &Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj),
        _ => Err(datetime_error("datetime: expected a datetime value")),
    }
}

pub(crate) fn datetime_format_from_value(value: &Value) -> String {
    match value {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => format_for_object(obj),
        _ => DEFAULT_DATETIME_FORMAT.to_string(),
    }
}

pub(crate) fn datetime_row_times_from_calendar_step(
    start: &Value,
    step: &Value,
    count: usize,
) -> BuiltinResult<Value> {
    let start_serials = serials_from_datetime_value(start)?;
    if start_serials.len() != 1 {
        return Err(datetime_error(
            "array2timetable: StartTime must be a scalar",
        ));
    }
    let (months, days) = calendar_duration_tensors_from_value(step)?;
    if months.len() != 1 || days.len() != 1 {
        return Err(datetime_error("array2timetable: TimeStep must be a scalar"));
    }
    let month_step = tensor::tensor_value_f64(&months, 0);
    let day_step = tensor::tensor_value_f64(&days, 0);
    if !month_step.is_finite() || !day_step.is_finite() {
        return Err(datetime_error("array2timetable: TimeStep must be finite"));
    }
    let one_month_step = Tensor::new(vec![month_step], vec![1, 1])
        .map_err(|error| datetime_error(format!("array2timetable: {error}")))?;
    let one_day_step = Tensor::new(vec![day_step], vec![1, 1])
        .map_err(|error| datetime_error(format!("array2timetable: {error}")))?;
    let (next_serial, _) = apply_calendar_duration_to_serials(
        &start_serials,
        &one_month_step,
        &one_day_step,
        1.0,
        "array2timetable",
    )?;
    if next_serial[0] <= tensor::tensor_value_f64(&start_serials, 0) {
        return Err(datetime_error("array2timetable: TimeStep must be positive"));
    }
    let mut serials = Vec::with_capacity(count);
    for index in 0..count {
        let factor = index as f64;
        let month_offset = Tensor::new(vec![month_step * factor], vec![1, 1])
            .map_err(|error| datetime_error(format!("array2timetable: {error}")))?;
        let day_offset = Tensor::new(vec![day_step * factor], vec![1, 1])
            .map_err(|error| datetime_error(format!("array2timetable: {error}")))?;
        let (value, _) = apply_calendar_duration_to_serials(
            &start_serials,
            &month_offset,
            &day_offset,
            1.0,
            "array2timetable",
        )?;
        serials.push(value[0]);
    }
    if count > 1 && serials.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(datetime_error(
            "array2timetable: TimeStep must produce increasing row times",
        ));
    }
    datetime_object_from_serial_tensor(
        Tensor::new(serials, vec![count, 1])
            .map_err(|error| datetime_error(format!("array2timetable: {error}")))?,
        datetime_format_from_value(start),
    )
}

pub fn datetime_string_array(value: &Value) -> BuiltinResult<Option<StringArray>> {
    let Value::Object(obj) = value else {
        return Ok(None);
    };
    if !obj.is_class(DATETIME_CLASS) {
        return Ok(None);
    }
    let serials = serial_tensor_for_object(obj)?;
    let format = format_for_object(obj);
    let values = tensor::tensor_values_f64_cow(&serials);
    let mut strings = Vec::with_capacity(values.len());
    for serial in values.iter().copied() {
        strings.push(format_serial(serial, &format)?);
    }
    let shape = tensor::default_shape_for(&serials.shape, values.len());
    let array = StringArray::new(strings, shape)
        .map_err(|err| datetime_error(format!("datetime: {err}")))?;
    Ok(Some(array))
}

pub fn datetime_display_text(value: &Value) -> BuiltinResult<Option<String>> {
    let Some(array) = datetime_string_array(value)? else {
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
            widths[col] = widths[col].max(array.data[idx].chars().count());
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
            let padding = widths[col].saturating_sub(text.chars().count());
            if padding > 0 {
                line.push_str(&" ".repeat(padding));
            }
        }
        lines.push(line);
    }
    Ok(Some(lines.join("\n")))
}
