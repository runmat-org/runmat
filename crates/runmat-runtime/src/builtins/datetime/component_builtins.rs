use super::capabilities::{
    DATETIME_CLASS, DATETIME_GPU_INPUT_EXTENSION, DATETIME_LOGICAL_INPUT_EXTENSION,
    DEFAULT_DATE_FORMAT, HOUR_TYPED_LEGACY_NUMERIC_EXTENSION,
    MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION, MONTH_TYPED_LEGACY_NUMERIC_EXTENSION,
    YEAR_TYPED_LEGACY_NUMERIC_EXTENSION,
};
use super::*;

#[runmat_macros::runtime_builtin(
    name = "year",
    descriptor(crate::builtins::datetime::DATETIME_YEAR_DESCRIPTOR),
    extensions(crate::builtins::datetime::YEAR_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::YEAR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract calendar year components from datetime values.",
    keywords = "year,datetime,date component"
)]
pub(super) async fn year_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (value, option) =
        prepare_legacy_component_input("year", value, &rest, &YEAR_TYPED_LEGACY_NUMERIC_EXTENSION)
            .await?;
    if is_datetime_object(&value) {
        let mode = option
            .as_deref()
            .unwrap_or("iso")
            .trim()
            .to_ascii_lowercase();
        return match mode.as_str() {
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
        };
    }
    numeric_component_from_modern_or_legacy("year", value, option.as_deref(), |naive| {
        naive.year() as f64
    })
}

#[runmat_macros::runtime_builtin(
    name = "month",
    descriptor(crate::builtins::datetime::DATETIME_MONTH_DESCRIPTOR),
    extensions(crate::builtins::datetime::MONTH_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::MONTH_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract month numbers from datetime arrays.",
    keywords = "month,datetime,date component"
)]
pub(super) async fn month_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (value, option) = prepare_legacy_component_input(
        "month",
        value,
        &rest,
        &MONTH_TYPED_LEGACY_NUMERIC_EXTENSION,
    )
    .await?;
    if is_datetime_object(&value) {
        let mode = option
            .as_deref()
            .unwrap_or("monthofyear")
            .trim()
            .to_ascii_lowercase();
        return match mode.as_str() {
            "monthofyear" => {
                component_tensor_from_datetime(&value, "month", |naive| naive.month() as f64)
            }
            "name" => month_names_from_datetime(&value, false),
            "shortname" => month_names_from_datetime(&value, true),
            _ => Err(datetime_error(format!(
                "month: unsupported month type '{mode}'"
            ))),
        };
    }
    numeric_component_from_modern_or_legacy("month", value, option.as_deref(), |naive| {
        naive.month() as f64
    })
}

#[runmat_macros::runtime_builtin(
    name = "day",
    descriptor(crate::builtins::datetime::DATETIME_DAY_DESCRIPTOR),
    extensions(crate::builtins::datetime::DAY_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::DAY_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract day numbers or names from datetime and legacy date inputs.",
    keywords = "day,datetime,date component,dayofweek,dayofyear"
)]
pub(super) async fn day_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
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
        .map_err(|err| datetime_error(format!("day: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
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
            let (serials, shape) = parse_legacy_day_text(value, legacy_input_format.as_deref())?;
            Tensor::new(serials, shape).map_err(|err| datetime_error(format!("day: {err}")))?
        }
        _ => serial_tensor_from_value(value, "day legacy serial input")?,
    };
    let shape = tensor::default_shape_for(&serials.shape, tensor::tensor_element_len(&serials));
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
                CharArray::new(text.chars().collect(), 1, text.chars().count())
                    .map_err(|err| datetime_error(format!("day: {err}")))?,
            ));
        }
        return Ok(Value::Cell(
            runmat_value::CellArray::new_with_shape(out, shape)
                .map_err(|err| datetime_error(format!("day: {err}")))?,
        ));
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
    tensor_or_scalar(out, shape)
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
    descriptor(crate::builtins::datetime::DATETIME_HOUR_DESCRIPTOR),
    extensions(crate::builtins::datetime::HOUR_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::HOUR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract hour components from datetime values.",
    keywords = "hour,datetime,time component"
)]
pub(super) async fn hour_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (value, format) =
        prepare_legacy_component_input("hour", value, &rest, &HOUR_TYPED_LEGACY_NUMERIC_EXTENSION)
            .await?;
    numeric_component_from_modern_or_legacy("hour", value, format.as_deref(), |naive| {
        naive.hour() as f64
    })
}

#[runmat_macros::runtime_builtin(
    name = "minute",
    descriptor(crate::builtins::datetime::DATETIME_MINUTE_DESCRIPTOR),
    extensions(crate::builtins::datetime::MINUTE_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::MINUTE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract minute numbers from datetime arrays.",
    keywords = "minute,datetime,time component"
)]
pub(super) async fn minute_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (value, format) = prepare_legacy_component_input(
        "minute",
        value,
        &rest,
        &MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION,
    )
    .await?;
    numeric_component_from_modern_or_legacy("minute", value, format.as_deref(), |naive| {
        naive.minute() as f64
    })
}

#[runmat_macros::runtime_builtin(
    name = "second",
    descriptor(crate::builtins::datetime::DATETIME_SECOND_DESCRIPTOR),
    integer_audit(crate::builtins::datetime::SECOND_INTEGER_AUDIT),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract second components from datetime values.",
    keywords = "second,datetime,time component"
)]
pub(super) async fn second_builtin(value: Value) -> crate::BuiltinResult<Value> {
    component_tensor_from_datetime(&value, "second", |naive| {
        naive.second() as f64 + f64::from(naive.nanosecond()) / 1_000_000_000.0
    })
}

#[runmat_macros::runtime_builtin(
    name = "isdatetime",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return true for datetime values.",
    keywords = "isdatetime,datetime,predicate"
)]
pub(super) fn isdatetime_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(is_datetime_object(&value)))
}

#[runmat_macros::runtime_builtin(
    name = "now",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return the current local date and time as a MATLAB serial date number.",
    keywords = "now,datenum,current time"
)]
pub(super) fn now_builtin() -> crate::BuiltinResult<Value> {
    Ok(Value::Num(datenum_from_naive(current_naive_local())))
}

#[runmat_macros::runtime_builtin(
    name = "today",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return the current local date as a datetime scalar.",
    keywords = "today,datetime,current date"
)]
pub(super) fn today_builtin() -> crate::BuiltinResult<Value> {
    let today = Local::now().date_naive().and_hms_opt(0, 0, 0).unwrap();
    datetime_from_date_only(today, DEFAULT_DATE_FORMAT)
}

#[runmat_macros::runtime_builtin(
    name = "clock",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return the current local date and time as a date vector.",
    keywords = "clock,datevec,current time"
)]
pub(super) fn clock_builtin() -> crate::BuiltinResult<Value> {
    let components = datevec_components_from_serial(datenum_from_naive(current_naive_local()))?;
    Ok(Value::Tensor(
        Tensor::new(components.to_vec(), vec![1, 6])
            .map_err(|err| datetime_error(format!("clock: {err}")))?,
    ))
}
