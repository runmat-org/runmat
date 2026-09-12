use super::capabilities::{
    DATETIME_CLASS, DATETIME_GPU_INPUT_EXTENSION, DATETIME_LOGICAL_INPUT_EXTENSION,
};
use super::*;

pub fn datetime_summary(value: &Value) -> BuiltinResult<Option<String>> {
    let Value::Object(obj) = value else {
        return Ok(None);
    };
    if !obj.is_class(DATETIME_CLASS) {
        return Ok(None);
    }
    let serials = serial_tensor_for_object(obj)?;
    if tensor::tensor_element_len(&serials) == 1 {
        return datetime_display_text(value);
    }
    let shape = tensor::default_shape_for(&serials.shape, tensor::tensor_element_len(&serials));
    Ok(Some(format!(
        "[{} datetime]",
        shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join("x")
    )))
}

pub(super) fn component_tensor_from_datetime(
    value: &Value,
    label: &str,
    extractor: impl Fn(&NaiveDateTime) -> f64,
) -> BuiltinResult<Value> {
    let serials = serials_from_datetime_value(value)?;
    let values = tensor::tensor_values_f64_cow(&serials);
    let mut out = Vec::with_capacity(values.len());
    for serial in values.iter().copied() {
        let naive = naive_from_datenum(serial)?;
        out.push(extractor(&naive));
    }
    if out.len() == 1 {
        Ok(Value::Num(out[0]))
    } else {
        let shape = tensor::default_shape_for(&serials.shape, values.len());
        let tensor =
            Tensor::new(out, shape).map_err(|err| datetime_error(format!("{label}: {err}")))?;
        Ok(Value::Tensor(tensor))
    }
}

pub(super) fn component_tensor_from_serials(
    serials: &Tensor,
    label: &str,
    extractor: impl Fn(&NaiveDateTime) -> f64,
) -> BuiltinResult<Value> {
    let values = tensor::tensor_values_f64_cow(serials);
    let mut out = Vec::with_capacity(values.len());
    for serial in values.iter().copied() {
        out.push(if serial.is_finite() {
            extractor(&naive_from_datenum(serial)?)
        } else {
            f64::NAN
        });
    }
    let shape = tensor::default_shape_for(&serials.shape, values.len());
    tensor_or_scalar(out, shape)
        .map_err(|err| datetime_error(format!("{label}: {}", err.message())))
}

pub(super) fn is_typed_legacy_numeric(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some() || tensor.as_f32_slice().is_some())
}

pub(super) async fn prepare_legacy_component_input(
    builtin: &'static str,
    value: Value,
    rest: &[Value],
    typed_extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<(Value, Option<String>)> {
    if rest.len() > 1 {
        return Err(datetime_error(format!(
            "{builtin}: expected at most one component type or legacy date format"
        )));
    }
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            builtin,
        )?;
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            builtin,
        )?;
    }
    if is_typed_legacy_numeric(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(typed_extension, builtin)?;
    }
    if rest
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Err(datetime_error(format!(
            "{builtin}: component type or legacy date format must be host text"
        )));
    }

    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("{builtin}: {}", err.message())))?;
    let option = rest
        .first()
        .map(|value| scalar_text(value, &format!("{builtin} component type or date format")))
        .transpose()?;
    Ok((value, option))
}

pub(super) fn numeric_component_from_modern_or_legacy(
    builtin: &'static str,
    value: Value,
    legacy_format: Option<&str>,
    extractor: impl Fn(&NaiveDateTime) -> f64,
) -> BuiltinResult<Value> {
    if is_datetime_object(&value) {
        if legacy_format.is_some() {
            return Err(datetime_error(format!(
                "{builtin}: legacy date format is not supported for datetime input"
            )));
        }
        return component_tensor_from_datetime(&value, builtin, extractor);
    }
    let serials = match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            let (serials, shape) = parse_legacy_component_text(value, legacy_format, builtin)?;
            Tensor::new(serials, shape)
                .map_err(|err| datetime_error(format!("{builtin}: {err}")))?
        }
        value => serial_tensor_from_value(value, &format!("{builtin} legacy serial input"))?,
    };
    component_tensor_from_serials(&serials, builtin, extractor)
}

pub(super) fn month_name(month: u32, short: bool) -> &'static str {
    const FULL: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const SHORT: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let index = month.saturating_sub(1) as usize;
    if short {
        SHORT.get(index).copied().unwrap_or("")
    } else {
        FULL.get(index).copied().unwrap_or("")
    }
}

pub(super) fn month_names_from_datetime(value: &Value, short: bool) -> BuiltinResult<Value> {
    let serials = serials_from_datetime_value(value)?;
    let values = tensor::tensor_values_f64_cow(&serials);
    let shape = tensor::default_shape_for(&serials.shape, values.len());
    let mut out = Vec::with_capacity(values.len());
    for serial in values.iter().copied() {
        let text = if serial.is_finite() {
            month_name(naive_from_datenum(serial)?.month(), short)
        } else {
            ""
        };
        out.push(Value::CharArray(CharArray::new_row(text)));
    }
    runmat_value::CellArray::new_with_shape(out, shape)
        .map(Value::Cell)
        .map_err(|err| datetime_error(format!("month: {err}")))
}

pub(super) fn tensor_or_scalar(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if data.len() == 1 {
        Ok(Value::Num(data[0]))
    } else {
        Ok(Value::Tensor(Tensor::new(data, shape).map_err(|err| {
            datetime_error(format!("datetime: {err}"))
        })?))
    }
}

pub(super) fn numeric_or_datetime_serial_tensor(
    value: Value,
    context: &str,
) -> BuiltinResult<Tensor> {
    match &value {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            let (serials, shape, _) = parse_text_input(value, None)?;
            Tensor::new(serials, shape).map_err(|err| datetime_error(format!("{context}: {err}")))
        }
        _ => serial_tensor_from_value(value, context),
    }
}

pub(super) fn datevec_components_from_serial(serial: f64) -> BuiltinResult<[f64; 6]> {
    let naive = naive_from_datenum(serial)?;
    Ok([
        naive.year() as f64,
        naive.month() as f64,
        naive.day() as f64,
        naive.hour() as f64,
        naive.minute() as f64,
        naive.second() as f64 + f64::from(naive.nanosecond()) / 1_000_000_000.0,
    ])
}

pub(super) fn datevec_matrix_from_serial_tensor(serials: &Tensor) -> BuiltinResult<Tensor> {
    let values = tensor::tensor_values_f64_cow(serials);
    let rows = values.len();
    let mut data = vec![0.0; rows.saturating_mul(6)];
    for (row, serial) in values.iter().enumerate() {
        let components = datevec_components_from_serial(*serial)?;
        for col in 0..6 {
            data[col * rows + row] = components[col];
        }
    }
    Tensor::new(data, vec![rows, 6]).map_err(|err| datetime_error(format!("datevec: {err}")))
}

pub(super) fn datetime_from_date_only(
    naive: NaiveDateTime,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    datetime_object_from_serials(vec![datenum_from_naive(naive)], vec![1, 1], format)
}

pub(super) fn current_naive_local() -> NaiveDateTime {
    Local::now().naive_local()
}

pub(super) fn days_in_month(year: i32, month: u32) -> BuiltinResult<u32> {
    let _ = NaiveDate::from_ymd_opt(year, month, 1)
        .ok_or_else(|| datetime_error("eomday: invalid year/month"))?;
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    let next = NaiveDate::from_ymd_opt(next_year, next_month, 1)
        .ok_or_else(|| datetime_error("eomday: invalid year/month"))?;
    Ok((next - Duration::days(1)).day())
}
