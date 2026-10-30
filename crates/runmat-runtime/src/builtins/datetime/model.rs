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

pub(super) fn datetime_utf16_error(source: runmat_value::Utf16DecodeError) -> RuntimeError {
    build_runtime_error("datetime: malformed UTF-16 character input")
        .with_builtin(BUILTIN_NAME)
        .with_identifier(
            DATETIME_ERROR_INVALID_INPUT
                .identifier
                .expect("datetime invalid-input descriptor identifier"),
        )
        .with_source(source)
        .build()
}

pub(super) fn datetime_argument_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(
            DATETIME_ERROR_INVALID_ARGUMENT
                .identifier
                .expect("datetime invalid-argument descriptor identifier"),
        )
        .build()
}

pub(super) fn datetime_execution_error(
    runtime: Option<&crate::context::RuntimeContext>,
    source: crate::execution::ExecutionServiceError,
) -> RuntimeError {
    let mut error = if source == crate::execution::ExecutionServiceError::Cancelled {
        build_runtime_error("datetime: execution was cancelled")
            .with_builtin(BUILTIN_NAME)
            .with_identifier(format!(
                "{}:Cancelled",
                runtime.map_or(
                    "RunMat".to_string(),
                    crate::context::RuntimeContext::error_namespace
                )
            ))
            .build()
    } else {
        source.clone().into_runtime_error()
    };
    error.source = Some(Box::new(source));
    error
}

pub(super) fn datetime_gather_error(
    runtime: Option<&crate::context::RuntimeContext>,
    source: RuntimeError,
) -> RuntimeError {
    let mut cause: &(dyn std::error::Error + 'static) = &source;
    let cancelled = loop {
        if matches!(
            cause.downcast_ref::<crate::execution::ExecutionServiceError>(),
            Some(crate::execution::ExecutionServiceError::Cancelled)
        ) {
            break true;
        }
        let Some(next) = cause.source() else {
            break false;
        };
        cause = next;
    };
    let mut error = if cancelled {
        build_runtime_error("datetime: execution was cancelled")
            .with_builtin(BUILTIN_NAME)
            .with_identifier(format!(
                "{}:Cancelled",
                runtime.map_or(
                    "RunMat".to_string(),
                    crate::context::RuntimeContext::error_namespace
                )
            ))
            .build()
    } else {
        datetime_error(format!("datetime: {}", source.message()))
    };
    error.source = Some(Box::new(source));
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
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
                default_value: Some(Value::string(DEFAULT_DATETIME_FORMAT.to_string())),
            },
        );

        let methods = crate::class_registry::standard_runtime_methods(DATETIME_CLASS);

        crate::class_registry::register_builtin_class(
            DATETIME_CLASS,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: DATETIME_CLASS.into(),
                parent: None,
                properties,
                methods,
            },
        );
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

        let methods = crate::class_registry::standard_runtime_methods(CALENDAR_DURATION_CLASS);

        crate::class_registry::register_builtin_class(
            CALENDAR_DURATION_CLASS,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: CALENDAR_DURATION_CLASS.into(),
                parent: None,
                properties,
                methods,
            },
        );
    });
}

pub(super) fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(nonmissing_text(text, context)?.to_owned()),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(nonmissing_text(&array.data()[0], context)?.to_owned())
        }
        Value::CharArray(array) if array.rows == 1 => {
            runmat_value::decode_utf16_code_units(array.data()).map_err(datetime_utf16_error)
        }
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

pub(super) fn parse_trailing_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(usize, DatetimeOptions)> {
    let mut positional_end = args.len();
    let mut options = DatetimeOptions::default();

    while positional_end >= 2 {
        let name = match scalar_text(&args[positional_end - 2], "option name") {
            Ok(text) => text,
            Err(_) => break,
        };
        let lowered = name.trim().to_ascii_lowercase();
        let value = scalar_text(&args[positional_end - 1], &format!("{name} option"))
            .map_err(|error| datetime_argument_error(error.message().to_owned()))?;
        let slot = match lowered.as_str() {
            "format" => &mut options.format,
            "convertfrom" => &mut options.convert_from,
            "inputformat" => &mut options.input_format,
            _ => {
                return Err(datetime_argument_error(format!(
                    "datetime: unsupported option '{name}'"
                )));
            }
        };
        if slot.replace(value).is_some() {
            return Err(datetime_argument_error(format!(
                "datetime: option '{name}' cannot be specified more than once"
            )));
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
    for index in 0..storage.len() {
        let value = storage
            .value_at(index)
            .expect("index is within integer storage");
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
    for index in 0..storage.len() {
        let value = storage
            .value_at(index)
            .expect("index is within integer storage");
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
    // Keep explicit empty and N-D numeric geometry; collapsing every empty
    // input to 0-by-1 would change the shape of date-array comparisons.
    let shape = if tensor.shape().is_empty() {
        vec![1, 1]
    } else {
        tensor.shape().to_vec()
    };
    let values = tensor::tensor_into_values_f64(tensor);
    Tensor::new(values, shape).map_err(|err| datetime_error(format!("datetime: {err}")))
}

pub(super) fn format_for_object(obj: &ObjectInstance) -> BuiltinResult<String> {
    Ok(match obj.properties().get(FORMAT_FIELD) {
        Some(Value::String(text)) => nonmissing_text(text, "Format")?.to_owned(),
        Some(Value::StringArray(array)) if array.data().len() == 1 => {
            nonmissing_text(&array.data()[0], "Format")?.to_owned()
        }
        Some(Value::CharArray(array)) if array.rows == 1 => {
            runmat_value::decode_utf16_code_units(array.data()).map_err(datetime_utf16_error)?
        }
        _ => DEFAULT_DATETIME_FORMAT.to_string(),
    })
}

pub(crate) fn serial_tensor_for_object(obj: &ObjectInstance) -> BuiltinResult<Tensor> {
    match obj.properties().get(SERIAL_FIELD) {
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
    runtime: Option<&crate::context::RuntimeContext>,
    serials: Tensor,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    ensure_datetime_class_registered();
    let fields = [
        (SERIAL_FIELD, Value::Tensor(serials)),
        (FORMAT_FIELD, Value::string(format.into())),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        DATETIME_CLASS.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
}

pub(super) fn datetime_object_from_serials(
    runtime: Option<&crate::context::RuntimeContext>,
    serials: Vec<f64>,
    shape: Vec<usize>,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    let tensor =
        Tensor::new(serials, shape).map_err(|err| datetime_error(format!("datetime: {err}")))?;
    datetime_object_from_serial_tensor(runtime, tensor, format)
}

pub(super) fn format_token_to_strftime(format: &str) -> String {
    crate::builtins::common::validation::datetime_format_to_strftime(format)
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
/// Retain logical argument generations through exact final-slot gathering.
pub(super) async fn gather_retained_args(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
        let gather = gather_if_needed_async(&args[index]);
        match runtime {
            Some(runtime) => runtime.scope(gather).await,
            None => gather.await,
        }
        .map_err(|err| datetime_error(format!("datetime: {}", err.message())))
    })
    .await
}
pub(super) async fn gather_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
) -> BuiltinResult<Value> {
    let gather = gather_if_needed_async(value);
    match runtime {
        Some(runtime) => runtime.scope(gather).await,
        None => gather.await,
    }
}

pub(super) fn nonmissing_text<'a>(
    text: &'a runmat_value::StringElement,
    context: &str,
) -> BuiltinResult<&'a str> {
    text.as_str()
        .ok_or_else(|| datetime_error(format!("datetime: {context} must not be missing")))
}
