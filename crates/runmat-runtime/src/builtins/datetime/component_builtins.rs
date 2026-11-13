use super::capabilities::{
    DATETIME_CLASS, DATETIME_GPU_INPUT_EXTENSION, DATETIME_LOGICAL_INPUT_EXTENSION,
    DEFAULT_DATE_FORMAT, HOUR_TYPED_LEGACY_NUMERIC_EXTENSION,
    MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION, MONTH_TYPED_LEGACY_NUMERIC_EXTENSION,
    YEAR_TYPED_LEGACY_NUMERIC_EXTENSION,
};
use super::*;

#[runmat_macros::runtime_builtin(
    name = "year",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn year_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if let Some(output) =
        map_legacy_component(runtime.as_ref(), ComponentKind::Year, &value, rest.view()).await?
    {
        return Ok(output);
    }
    let work = super::allocation::UnboundedCalendarWork::admit(runtime.as_ref(), "year")?;
    let (value, option) = prepare_legacy_component_input(
        &work,
        "year",
        value,
        rest.view(),
        &YEAR_TYPED_LEGACY_NUMERIC_EXTENSION,
    )
    .await?;
    if is_datetime_object(&value) {
        let mode = option
            .as_deref()
            .unwrap_or("iso")
            .trim()
            .to_ascii_lowercase();
        return work.finish(match mode.as_str() {
            "iso" => component_tensor_from_datetime(&value, "year", |naive| naive.year() as f64),
            "gregorian" => component_tensor_from_datetime(&value, "year", |naive| {
                let year = naive.year();
                if year > 0 {
                    year as f64
                } else {
                    f64::from(1 - year)
                }
            }),
            _ => Err(datetime_error(format!(
                "year: unsupported year type '{mode}'"
            ))),
        });
    }
    work.finish(numeric_component_from_modern_or_legacy(
        runtime.as_ref(),
        "year",
        value,
        option.as_deref(),
        |naive| naive.year() as f64,
    ))
}

#[runmat_macros::runtime_builtin(
    name = "month",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn month_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if let Some(output) =
        map_legacy_component(runtime.as_ref(), ComponentKind::Month, &value, rest.view()).await?
    {
        return Ok(output);
    }
    let work = super::allocation::UnboundedCalendarWork::admit(runtime.as_ref(), "month")?;
    let (value, option) = prepare_legacy_component_input(
        &work,
        "month",
        value,
        rest.view(),
        &MONTH_TYPED_LEGACY_NUMERIC_EXTENSION,
    )
    .await?;
    if is_datetime_object(&value) {
        let mode = option
            .as_deref()
            .unwrap_or("monthofyear")
            .trim()
            .to_ascii_lowercase();
        return work.finish(match mode.as_str() {
            "monthofyear" => {
                component_tensor_from_datetime(&value, "month", |naive| naive.month() as f64)
            }
            "name" => month_names_from_datetime(&value, false),
            "shortname" => month_names_from_datetime(&value, true),
            _ => Err(datetime_error(format!(
                "month: unsupported month type '{mode}'"
            ))),
        });
    }
    work.finish(numeric_component_from_modern_or_legacy(
        runtime.as_ref(),
        "month",
        value,
        option.as_deref(),
        |naive| naive.month() as f64,
    ))
}

#[runmat_macros::runtime_builtin(
    name = "day",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn day_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if let Some(output) =
        map_legacy_component(runtime.as_ref(), ComponentKind::Day, &value, rest.view()).await?
    {
        return Ok(output);
    }
    let work = super::allocation::UnboundedCalendarWork::admit(runtime.as_ref(), "day")?;
    if std::iter::once(&value)
        .chain(rest.iter())
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            "day",
        )?;
    }
    if std::iter::once(&value)
        .chain(rest.iter())
        .any(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            "day",
        )?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| crate::builtins::common::map_control_flow_with_builtin(err, "day"))?;
    work.check()?;
    let mut budget = runtime
        .as_ref()
        .map(crate::context::RuntimeContext::cpu_cooperation);
    let rest = rest
        .try_map_preserving_roots_async(rest.len(), runtime.as_ref(), async |index| {
            work.check()?;
            let value = crate::dispatcher::gather_if_needed_owned_async(
                rest[index].clone(),
                budget.as_mut(),
            )
            .await
            .map_err(|err| datetime_gather_error(runtime.as_ref(), err))?;
            work.check()?;
            Ok(value)
        })
        .await?;
    work.check()?;
    if rest.len() > 1 {
        return Err(datetime_error("day: expected at most one day type"));
    }
    let modern_datetime = is_datetime_object(&value);
    let mode = if modern_datetime {
        rest.first()
            .map(|value| scalar_text(value, "day type"))
            .transpose()?
            .unwrap_or_else(|| "dayofmonth".to_string())
            .trim()
            .to_ascii_lowercase()
    } else {
        "dayofmonth".to_string()
    };
    let legacy_input_format = if modern_datetime {
        None
    } else {
        rest.first()
            .map(|value| scalar_text(value, "legacy input format"))
            .transpose()?
    };
    let serials = match &value {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj)?,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            let (serials, shape) =
                parse_legacy_day_text(runtime.as_ref(), value, legacy_input_format.as_deref())?;
            Tensor::new(serials, shape).map_err(|err| datetime_error(format!("day: {err}")))?
        }
        _ => serial_tensor_from_value(value, "day legacy serial input")?,
    };
    let shape = tensor::default_shape_for(serials.shape(), tensor::tensor_element_len(&serials));
    let values = tensor::tensor_values_f64_cow(&serials);
    if matches!(mode.as_str(), "name" | "shortname") {
        let short = mode == "shortname";
        let mut out = Vec::with_capacity(values.len());
        for serial in values.iter().copied() {
            let text = if serial.is_finite() {
                let weekday = naive_from_datenum(serial)?.weekday();
                weekday_name(weekday, short).to_string()
            } else {
                String::new()
            };
            out.push(Value::CharArray(
                CharArray::from_unicode_scalars(
                    text.chars().collect(),
                    1,
                    text.encode_utf16().count(),
                )
                .map_err(|err| datetime_error(format!("day: {err}")))?,
            ));
        }
        return work.finish(Ok(Value::Cell(
            runmat_value::CellArray::new_with_shape(out, shape)
                .map_err(|err| datetime_error(format!("day: {err}")))?,
        )));
    }
    let extractor: fn(&NaiveDateTime) -> f64 = match mode.as_str() {
        "dayofmonth" => |value| value.day() as f64,
        "dayofweek" => |value| f64::from(value.weekday().num_days_from_sunday() + 1),
        "iso-dayofweek" => |value| f64::from(value.weekday().number_from_monday()),
        "dayofyear" => |value| f64::from(value.ordinal()),
        _ => {
            return Err(datetime_error(format!(
                "day: unsupported day type '{mode}'"
            )))
        }
    };
    let mut out = Vec::with_capacity(values.len());
    for serial in values.iter().copied() {
        out.push(if serial.is_finite() {
            extractor(&naive_from_datenum(serial)?)
        } else {
            f64::NAN
        });
    }
    work.finish(tensor_or_scalar(out, shape))
}

pub(super) fn weekday_name(weekday: Weekday, short: bool) -> &'static str {
    match (weekday, short) {
        (Weekday::Mon, false) => "Monday",
        (Weekday::Tue, false) => "Tuesday",
        (Weekday::Wed, false) => "Wednesday",
        (Weekday::Thu, false) => "Thursday",
        (Weekday::Fri, false) => "Friday",
        (Weekday::Sat, false) => "Saturday",
        (Weekday::Sun, false) => "Sunday",
        (Weekday::Mon, true) => "Mon",
        (Weekday::Tue, true) => "Tue",
        (Weekday::Wed, true) => "Wed",
        (Weekday::Thu, true) => "Thu",
        (Weekday::Fri, true) => "Fri",
        (Weekday::Sat, true) => "Sat",
        (Weekday::Sun, true) => "Sun",
    }
}

#[runmat_macros::runtime_builtin(
    name = "hour",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn hour_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if let Some(output) =
        map_legacy_component(runtime.as_ref(), ComponentKind::Hour, &value, rest.view()).await?
    {
        return Ok(output);
    }
    let work = super::allocation::UnboundedCalendarWork::admit(runtime.as_ref(), "hour")?;
    let (value, format) = prepare_legacy_component_input(
        &work,
        "hour",
        value,
        rest.view(),
        &HOUR_TYPED_LEGACY_NUMERIC_EXTENSION,
    )
    .await?;
    work.finish(numeric_component_from_modern_or_legacy(
        runtime.as_ref(),
        "hour",
        value,
        format.as_deref(),
        |naive| naive.hour() as f64,
    ))
}

#[runmat_macros::runtime_builtin(
    name = "minute",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn minute_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if let Some(output) =
        map_legacy_component(runtime.as_ref(), ComponentKind::Minute, &value, rest.view()).await?
    {
        return Ok(output);
    }
    let work = super::allocation::UnboundedCalendarWork::admit(runtime.as_ref(), "minute")?;
    let (value, format) = prepare_legacy_component_input(
        &work,
        "minute",
        value,
        rest.view(),
        &MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION,
    )
    .await?;
    work.finish(numeric_component_from_modern_or_legacy(
        runtime.as_ref(),
        "minute",
        value,
        format.as_deref(),
        |naive| naive.minute() as f64,
    ))
}

#[runmat_macros::runtime_builtin(
    name = "isdatetime",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::datetime"
)]
pub(super) fn isdatetime_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    super::operators::remaining::predicate(runtime.as_ref(), args.view())
}

#[runmat_macros::runtime_builtin(
    name = "now",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) fn now_builtin(
    runtime: Option<crate::context::RuntimeContext>,
) -> crate::BuiltinResult<Value> {
    Ok(Value::Num(datenum_from_naive(current_naive_local(
        runtime.as_ref(),
    )?)))
}

#[runmat_macros::runtime_builtin(
    name = "today",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) fn today_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let output_type = match args.len() {
        0 => runmat_builtins::TodayOutputType::Datenum,
        1 => {
            let option = scalar_text(&args[0], "today outputType")?;
            runmat_builtins::TodayOutputType::parse(&option).ok_or_else(|| {
                datetime_argument_error("today: outputType must be 'datenum' or 'datetime'")
            })?
        }
        _ => {
            return Err(datetime_argument_error(
                "today: expected at most one outputType",
            ))
        }
    };
    // One current-time read per successful invocation, after option validation.
    let today = current_naive_local(runtime.as_ref())?
        .date()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let serial = datenum_from_naive(today);
    match output_type {
        runmat_builtins::TodayOutputType::Datenum => Ok(Value::Num(serial)),
        runmat_builtins::TodayOutputType::Datetime => datetime_object_from_serials(
            runtime.as_ref(),
            vec![serial],
            vec![1, 1],
            DEFAULT_DATE_FORMAT,
        ),
    }
}

#[runmat_macros::runtime_builtin(
    name = "clock",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) fn clock_builtin(
    runtime: Option<crate::context::RuntimeContext>,
) -> crate::BuiltinResult<Value> {
    let components =
        datevec_components_from_serial(datenum_from_naive(current_naive_local(runtime.as_ref())?))?;
    super::allocation::clock_components_output(runtime.as_ref(), components)
}
