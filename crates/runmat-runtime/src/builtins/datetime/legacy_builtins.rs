use super::capabilities::{
    BUILTIN_NAME, DATETIME_CLASS, DATETIME_GPU_INPUT_EXTENSION, DATETIME_LOGICAL_INPUT_EXTENSION,
    DEFAULT_DATETIME_FORMAT, SECONDS_PER_DAY,
};
use super::*;
use runmat_builtins::{
    DATENUM_PARTIAL_COMPONENT_EXTENSION, ISBETWEEN_SERIAL_DATE_EXTENSION,
    LEGACY_TYPED_NUMERIC_EXTENSION,
};

fn preflight_legacy_inputs<'a>(
    builtin: &'static str,
    mut values: impl Iterator<Item = &'a Value> + Clone,
    typed_numeric_is_extension: bool,
) -> BuiltinResult<()> {
    if values
        .clone()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            builtin,
        )?;
    }
    if values
        .clone()
        .any(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            builtin,
        )?;
    }
    if typed_numeric_is_extension && values.any(is_typed_legacy_numeric) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LEGACY_TYPED_NUMERIC_EXTENSION,
            builtin,
        )?;
    }
    Ok(())
}

#[runmat_macros::runtime_builtin(
    name = "datenum",
    builtin_path = "crate::builtins::datetime",
    binding_variant = "default"
)]
pub(super) async fn datenum_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !matches!(args.len(), 0 | 1 | 3..=6) {
        return Err(datetime_error(
            "datenum: expected datetime, text, date vector, or Y/M/D components",
        ));
    }
    if matches!(args.len(), 4 | 5) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATENUM_PARTIAL_COMPONENT_EXTENSION,
            "datenum",
        )?;
    }
    preflight_legacy_inputs("datenum", args.iter(), true)?;
    let args = gather_args(&args).await?;
    let tensor = match args.len() {
        0 => Tensor::new(vec![datenum_from_naive(current_naive_local())], vec![1, 1])
            .map_err(|err| datetime_error(format!("datenum: {err}")))?,
        1 => match &args[0] {
            Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj)?,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                let (serials, shape, _) = parse_text_input(args[0].clone(), None)?;
                Tensor::new(serials, shape)
                    .map_err(|err| datetime_error(format!("datenum: {err}")))?
            }
            Value::Tensor(_) => {
                if let Ok(datevec) = tensor_from_datevec_like(args[0].clone(), "datenum") {
                    datenum_from_datevec_tensor(&datevec, "datenum")?
                } else {
                    serial_tensor_from_value(args[0].clone(), "datenum")?
                }
            }
            _ => serial_tensor_from_value(args[0].clone(), "datenum")?,
        },
        3..=6 => {
            let datetime = build_from_components(args, None)?;
            serials_from_datetime_value(&datetime)?
        }
        _ => {
            return Err(datetime_error(
                "datenum: expected datetime, text, date vector, or Y/M/D components",
            ))
        }
    };
    if tensor::tensor_element_len(&tensor) == 1 {
        Ok(Value::Num(tensor::tensor_value_f64(&tensor, 0)))
    } else {
        Ok(Value::Tensor(tensor))
    }
}

#[runmat_macros::runtime_builtin(
    name = "datevec",
    builtin_path = "crate::builtins::datetime",
    binding_variant = "default"
)]
pub(super) async fn datevec_builtin(value: Value) -> crate::BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 6) {
        return Err(datetime_error("datevec: supports at most six outputs"));
    }
    preflight_legacy_inputs("datevec", [&value].into_iter(), true)?;
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("datevec: {}", err.message())))?;
    let serials = numeric_or_datetime_serial_tensor(value, "datevec")?;
    let matrix = datevec_matrix_from_serial_tensor(&serials)?;
    if let Some(out_count) = crate::output_count::current_output_count().filter(|count| *count != 1)
    {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let mut outputs = Vec::with_capacity(out_count.min(6));
        let matrix_values = tensor::tensor_values_f64_cow(&matrix);
        for col in 0..6.min(out_count) {
            let mut data = Vec::with_capacity(matrix.rows);
            for row in 0..matrix.rows {
                data.push(matrix_values[col * matrix.rows + row]);
            }
            outputs.push(if data.len() == 1 {
                Value::Num(data[0])
            } else {
                Value::Tensor(
                    Tensor::new(data, vec![matrix.rows, 1])
                        .map_err(|err| datetime_error(format!("datevec: {err}")))?,
                )
            });
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }
    Ok(Value::Tensor(matrix))
}

#[runmat_macros::runtime_builtin(
    name = "datestr",
    builtin_path = "crate::builtins::datetime",
    binding_variant = "default"
)]
pub(super) async fn datestr_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(datetime_error(
            "datestr: expected at most one format argument",
        ));
    }
    preflight_legacy_inputs("datestr", [&value].into_iter(), true)?;
    if rest
        .first()
        .is_some_and(|option| matches!(option, Value::GpuTensor(_)))
    {
        return Err(datetime_error("datestr: format must be host text"));
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("datestr: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    let format = rest
        .first()
        .map(|value| scalar_text(value, "datestr format"))
        .transpose()?
        .unwrap_or_else(|| DEFAULT_DATETIME_FORMAT.to_string());
    let serials = match &value {
        Value::Tensor(_) => {
            if let Ok(datevec) = tensor_from_datevec_like(value.clone(), "datestr") {
                datenum_from_datevec_tensor(&datevec, "datestr")?
            } else {
                numeric_or_datetime_serial_tensor(value, "datestr")?
            }
        }
        _ => numeric_or_datetime_serial_tensor(value, "datestr")?,
    };
    let serial_values = tensor::tensor_values_f64_cow(&serials);
    let mut rows = Vec::with_capacity(serial_values.len());
    for serial in serial_values.iter() {
        rows.push(format_serial(*serial, &format)?);
    }
    Ok(Value::CharArray(char_array_from_rows(&rows, "datestr")?))
}

#[runmat_macros::runtime_builtin(
    name = "weekday",
    builtin_path = "crate::builtins::datetime",
    binding_variant = "default"
)]
pub(super) async fn weekday_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(datetime_error(
            "weekday: expected a date and at most one day-name form",
        ));
    }
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        return Err(datetime_error("weekday: supports at most two outputs"));
    }
    let long_names = match rest.first() {
        None => false,
        Some(value) => match scalar_text(value, "weekday DayForm")?
            .to_ascii_lowercase()
            .as_str()
        {
            "short" => false,
            "long" => true,
            _ => return Err(datetime_error("weekday: DayForm must be 'short' or 'long'")),
        },
    };
    preflight_legacy_inputs("weekday", [&value].into_iter(), false)?;
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("weekday: {}", err.message())))?;
    let serials = numeric_or_datetime_serial_tensor(value, "weekday")?;
    let serial_values = tensor::tensor_values_f64_cow(&serials);
    let mut nums = Vec::with_capacity(serial_values.len());
    let include_names = crate::output_count::current_output_count() == Some(2);
    let mut names = if include_names {
        Vec::with_capacity(serial_values.len())
    } else {
        Vec::new()
    };
    for serial in serial_values.iter() {
        let weekday = naive_from_datenum(*serial)?.weekday();
        nums.push(f64::from(weekday.num_days_from_sunday()) + 1.0);
        if include_names {
            names.push(
                match (weekday, long_names) {
                    (Weekday::Sun, true) => "Sunday",
                    (Weekday::Mon, true) => "Monday",
                    (Weekday::Tue, true) => "Tuesday",
                    (Weekday::Wed, true) => "Wednesday",
                    (Weekday::Thu, true) => "Thursday",
                    (Weekday::Fri, true) => "Friday",
                    (Weekday::Sat, true) => "Saturday",
                    (Weekday::Sun, false) => "Sun",
                    (Weekday::Mon, false) => "Mon",
                    (Weekday::Tue, false) => "Tues",
                    (Weekday::Wed, false) => "Wed",
                    (Weekday::Thu, false) => "Thurs",
                    (Weekday::Fri, false) => "Fri",
                    (Weekday::Sat, false) => "Sat",
                }
                .to_string(),
            );
        }
    }
    let shape = tensor::default_shape_for(&serials.shape, serial_values.len());
    let num_value = tensor_or_scalar(nums, shape.clone())?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count < 2 {
            return Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![num_value],
            ));
        }
        let name_value = Value::CharArray(char_array_from_rows(&names, "weekday")?);
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![num_value, name_value],
        ));
    }
    Ok(num_value)
}

#[runmat_macros::runtime_builtin(
    name = "eomday",
    builtin_path = "crate::builtins::datetime",
    binding_variant = "default"
)]
pub(super) async fn eomday_builtin(year: Value, month: Value) -> crate::BuiltinResult<Value> {
    preflight_legacy_inputs("eomday", [&year, &month].into_iter(), false)?;
    let year = gather_if_needed_async(&year)
        .await
        .map_err(|err| datetime_error(format!("eomday: {}", err.message())))?;
    let month = gather_if_needed_async(&month)
        .await
        .map_err(|err| datetime_error(format!("eomday: {}", err.message())))?;
    let years = component_tensor(year, "year")?;
    let months = component_tensor(month, "month")?;
    let (year_data, month_data, shape) =
        tensor::binary_numeric_tensors(&years, &months, "eomday", BUILTIN_NAME)?;
    let mut out = Vec::with_capacity(year_data.len());
    for (year, month) in year_data.iter().zip(month_data.iter()) {
        let year = round_component(*year, "year", -262_000, 262_000)? as i32;
        let month = round_component(*month, "month", 1, 12)? as u32;
        out.push(f64::from(days_in_month(year, month)?));
    }
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "etime",
    builtin_path = "crate::builtins::datetime",
    binding_variant = "default"
)]
pub(super) async fn etime_builtin(t2: Value, t1: Value) -> crate::BuiltinResult<Value> {
    preflight_legacy_inputs("etime", [&t2, &t1].into_iter(), true)?;
    let t2 = gather_if_needed_async(&t2)
        .await
        .map_err(|err| datetime_error(format!("etime: {}", err.message())))?;
    let t1 = gather_if_needed_async(&t1)
        .await
        .map_err(|err| datetime_error(format!("etime: {}", err.message())))?;
    let t2 = datenum_from_datevec_tensor(&tensor_from_datevec_like(t2, "etime")?, "etime")?;
    let t1 = datenum_from_datevec_tensor(&tensor_from_datevec_like(t1, "etime")?, "etime")?;
    let (left, right, shape) = tensor::binary_numeric_tensors(&t2, &t1, "etime", BUILTIN_NAME)?;
    let out = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b) * SECONDS_PER_DAY)
        .collect::<Vec<_>>();
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "isbetween",
    builtin_path = "crate::builtins::datetime",
    binding_variant = "default"
)]
pub(super) async fn isbetween_builtin(
    value: Value,
    lower: Value,
    upper: Value,
) -> crate::BuiltinResult<Value> {
    if [&value, &lower, &upper].iter().any(|operand| {
        matches!(
            operand,
            Value::Num(_)
                | Value::Int(_)
                | Value::Tensor(_)
                | Value::String(_)
                | Value::StringArray(_)
                | Value::CharArray(_)
        )
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ISBETWEEN_SERIAL_DATE_EXTENSION,
            "isbetween",
        )?;
    }
    preflight_legacy_inputs("isbetween", [&value, &lower, &upper].into_iter(), false)?;
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("isbetween: {}", err.message())))?;
    let lower = gather_if_needed_async(&lower)
        .await
        .map_err(|err| datetime_error(format!("isbetween: {}", err.message())))?;
    let upper = gather_if_needed_async(&upper)
        .await
        .map_err(|err| datetime_error(format!("isbetween: {}", err.message())))?;
    let values = numeric_or_datetime_serial_tensor(value, "isbetween")?;
    let lower = numeric_or_datetime_serial_tensor(lower, "isbetween")?;
    let upper = numeric_or_datetime_serial_tensor(upper, "isbetween")?;
    let (values_data, lower_data, upper_data, shape) =
        implicit_broadcast_three_numeric_tensors(&values, &lower, &upper, "isbetween")?;
    let out = values_data
        .iter()
        .zip(lower_data.iter())
        .zip(upper_data.iter())
        .map(|((value, lower), upper)| u8::from(value >= lower && value <= upper))
        .collect::<Vec<_>>();
    if out.len() == 1 {
        Ok(Value::Bool(out[0] != 0))
    } else {
        Ok(Value::LogicalArray(
            runmat_value::LogicalArray::new(out, shape)
                .map_err(|err| datetime_error(format!("isbetween: {err}")))?,
        ))
    }
}
