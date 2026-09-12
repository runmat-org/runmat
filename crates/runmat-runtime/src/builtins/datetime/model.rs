use super::capabilities::{
    BUILTIN_NAME, CALENDAR_DAYS_FIELD, CALENDAR_DURATION_CLASS, CALENDAR_MONTHS_FIELD,
    DATETIME_CLASS, DATETIME_LOGICAL_INPUT_EXTENSION, DEFAULT_DATETIME_FORMAT, FORMAT_FIELD,
    SECONDS_PER_DAY, SERIAL_FIELD, UNIX_DATENUM,
};
use super::*;

pub(super) fn datetime_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(
            DATETIME_ERROR_INVALID_INPUT
                .identifier
                .expect("datetime invalid-input descriptor identifier"),
        )
        .build()
}

pub(super) fn ensure_datetime_class_registered() {
    DATETIME_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        properties.insert(
            FORMAT_FIELD.into(),
            crate::class_registry::RuntimeProperty {
                name: FORMAT_FIELD.into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: Some(Value::String(DEFAULT_DATETIME_FORMAT.to_string())),
            },
        );

        let mut methods = HashMap::new();
        for name in [
            OBJECT_SUBSREF_METHOD,
            OBJECT_SUBSASGN_METHOD,
            runmat_types::StaticMethodName::new("plus"),
            runmat_types::StaticMethodName::new("minus"),
            runmat_types::StaticMethodName::new("eq"),
            runmat_types::StaticMethodName::new("ne"),
            runmat_types::StaticMethodName::new("lt"),
            runmat_types::StaticMethodName::new("le"),
            runmat_types::StaticMethodName::new("gt"),
            runmat_types::StaticMethodName::new("ge"),
        ] {
            methods.insert(
                name.into(),
                crate::class_registry::RuntimeMethod {
                    name: name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: format!("{DATETIME_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            );
        }

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: DATETIME_CLASS.into(),
            parent: None,
            properties,
            methods,
        });
    });
}

pub(super) fn ensure_calendar_duration_class_registered() {
    CALENDAR_DURATION_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in [CALENDAR_MONTHS_FIELD, CALENDAR_DAYS_FIELD] {
            properties.insert(
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Public,
                    default_value: Some(Value::Num(0.0)),
                },
            );
        }

        let mut methods = HashMap::new();
        for name in [
            runmat_types::StaticMethodName::new("plus"),
            runmat_types::StaticMethodName::new("minus"),
            runmat_types::StaticMethodName::new("eq"),
            runmat_types::StaticMethodName::new("ne"),
        ] {
            methods.insert(
                name.into(),
                crate::class_registry::RuntimeMethod {
                    name: name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: format!("{CALENDAR_DURATION_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            );
        }

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: CALENDAR_DURATION_CLASS.into(),
            parent: None,
            properties,
            methods,
        });
    });
}

pub(super) async fn gather_args(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(
            gather_if_needed_async(arg)
                .await
                .map_err(|err| datetime_error(format!("datetime: {}", err.message())))?,
        );
    }
    Ok(out)
}

pub(super) fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        _ => Err(datetime_error(format!(
            "datetime: {context} must be a string scalar or character vector"
        ))),
    }
}

#[derive(Default)]
pub(super) struct DatetimeOptions {
    pub(super) format: Option<String>,
    pub(super) convert_from: Option<String>,
    pub(super) input_format: Option<String>,
}

pub(super) fn parse_trailing_options(args: &[Value]) -> BuiltinResult<(usize, DatetimeOptions)> {
    let mut positional_end = args.len();
    let mut options = DatetimeOptions::default();

    while positional_end >= 2 {
        let name = match scalar_text(&args[positional_end - 2], "option name") {
            Ok(text) => text,
            Err(_) => break,
        };
        let lowered = name.trim().to_ascii_lowercase();
        let value = scalar_text(&args[positional_end - 1], &format!("{name} option"))?;
        match lowered.as_str() {
            "format" => options.format = Some(value),
            "convertfrom" => options.convert_from = Some(value),
            "inputformat" => options.input_format = Some(value),
            _ => break,
        }
        positional_end -= 2;
    }

    Ok((positional_end, options))
}

pub(super) fn tensor_from_numeric(value: Value, context: &str) -> BuiltinResult<Tensor> {
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    tensor::value_into_tensor_for(context, value)
        .map_err(|message| datetime_error(format!("datetime: {message}")))
}

pub(super) fn validate_authoritative_integer_storage(
    tensor: &Tensor,
    context: &str,
) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    for value in storage.exact_values() {
        let Some(value) = value.try_to_i64() else {
            return Err(datetime_error(format!(
                "datetime: {context} integer value is outside the supported calendar range"
            )));
        };
        // Chrono's civil calendar is far narrower than i64. Reject from exact
        // storage while the value is still authoritative; every admitted
        // integer is then exactly representable at the serial-date boundary.
        if !(-1_000_000_000..=1_000_000_000).contains(&value) {
            return Err(datetime_error(format!(
                "datetime: {context} integer value is outside the supported calendar range"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_authoritative_serial_storage(
    tensor: &Tensor,
    context: &str,
) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    for value in storage.exact_values() {
        let Some(value) = value.try_to_i64() else {
            return Err(datetime_error(format!(
                "datetime: {context} integer serial is outside the supported serial-date range"
            )));
        };
        // RunMat's current serial-date representation and Chrono-backed civil
        // calendar are narrower than i64. Establish that boundary from exact
        // storage so no wide I64/U64 is rounded into apparent admissibility.
        if !(-1_000_000_000..=1_000_000_000).contains(&value) {
            return Err(datetime_error(format!(
                "datetime: {context} integer serial is outside the supported serial-date range"
            )));
        }
    }
    Ok(())
}

pub(super) fn serial_tensor_from_value(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor_from_numeric(value, context)?;
    validate_authoritative_serial_storage(&tensor, context)?;
    let tensor = tensor::integer_tensor_to_f64(tensor)
        .map_err(|err| datetime_error(format!("datetime: {err}")))?;
    let shape = tensor::default_shape_for(&tensor.shape, tensor::tensor_element_len(&tensor));
    let values = tensor::tensor_into_values_f64(tensor);
    Tensor::new(values, shape).map_err(|err| datetime_error(format!("datetime: {err}")))
}

pub(super) fn format_for_object(obj: &ObjectInstance) -> String {
    match obj.properties.get(FORMAT_FIELD) {
        Some(Value::String(text)) => text.clone(),
        Some(Value::StringArray(array)) if array.data.len() == 1 => array.data[0].clone(),
        Some(Value::CharArray(array)) if array.rows == 1 => array.data.iter().collect(),
        _ => DEFAULT_DATETIME_FORMAT.to_string(),
    }
}

pub(crate) fn serial_tensor_for_object(obj: &ObjectInstance) -> BuiltinResult<Tensor> {
    match obj.properties.get(SERIAL_FIELD) {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        Some(Value::Num(value)) => Tensor::new(vec![*value], vec![1, 1])
            .map_err(|err| datetime_error(format!("datetime: {err}"))),
        Some(other) => Err(datetime_error(format!(
            "datetime: invalid internal serial storage {other:?}"
        ))),
        None => Err(datetime_error("datetime: missing internal serial storage")),
    }
}

pub(crate) fn datetime_object_from_serial_tensor(
    serials: Tensor,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    ensure_datetime_class_registered();
    let mut object = ObjectInstance::new(DATETIME_CLASS.to_string());
    object
        .properties
        .insert(SERIAL_FIELD.to_string(), Value::Tensor(serials));
    object
        .properties
        .insert(FORMAT_FIELD.to_string(), Value::String(format.into()));
    Ok(Value::Object(object))
}

pub(super) fn datetime_object_from_serials(
    serials: Vec<f64>,
    shape: Vec<usize>,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    let tensor =
        Tensor::new(serials, shape).map_err(|err| datetime_error(format!("datetime: {err}")))?;
    datetime_object_from_serial_tensor(tensor, format)
}

pub(super) fn format_token_to_strftime(format: &str) -> String {
    let mut out = format.to_string();
    for (src, dst) in [
        ("yyyy", "%Y"),
        ("MMM", "%b"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("HH", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
    ] {
        out = out.replace(src, dst);
    }
    out
}

pub(crate) fn datenum_from_naive(datetime: NaiveDateTime) -> f64 {
    let base = NaiveDate::from_ymd_opt(1970, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let duration = datetime - base;
    let seconds = duration.num_seconds();
    let nanos = (duration - Duration::seconds(seconds))
        .num_nanoseconds()
        .unwrap_or(0);
    let total_seconds = seconds as f64 + nanos as f64 / 1_000_000_000.0;
    total_seconds / SECONDS_PER_DAY + UNIX_DATENUM
}

pub(crate) fn naive_from_datenum(serial: f64) -> BuiltinResult<NaiveDateTime> {
    if !serial.is_finite() {
        return Err(datetime_error(
            "datetime: serial date numbers must be finite",
        ));
    }
    let total_seconds = (serial - UNIX_DATENUM) * SECONDS_PER_DAY;
    if !total_seconds.is_finite()
        || total_seconds < i64::MIN as f64
        || total_seconds > i64::MAX as f64
    {
        return Err(datetime_error(
            "datetime: serial date number is outside the supported range",
        ));
    }
    let mut seconds = total_seconds.floor() as i64;
    let mut nanos = ((total_seconds - seconds as f64) * 1_000_000_000.0).round() as i64;
    if nanos == 1_000_000_000 {
        seconds = seconds.checked_add(1).ok_or_else(|| {
            datetime_error("datetime: serial date number is outside the supported range")
        })?;
        nanos = 0;
    }
    let base = NaiveDate::from_ymd_opt(1970, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let duration = Duration::try_seconds(seconds)
        .and_then(|duration| duration.checked_add(&Duration::nanoseconds(nanos)))
        .ok_or_else(|| {
            datetime_error("datetime: serial date number is outside the supported range")
        })?;
    base.checked_add_signed(duration).ok_or_else(|| {
        datetime_error("datetime: serial date number is outside the supported range")
    })
}
