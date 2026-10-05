use super::capabilities::{
    BUILTIN_NAME, DATETIME_CLASS, DATETIME_GPU_INPUT_EXTENSION,
    DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION, DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_RAW_DATENUM_EXTENSION, DEFAULT_DATETIME_FORMAT,
};
use super::*;

#[runmat_macros::runtime_builtin(
    name = "datetime",
    binding_variant = "default",
    builtin_path = "crate::builtins::datetime"
)]
pub(super) async fn datetime_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_datetime_class_registered();
    let (positional_end, options) = parse_trailing_options(&args)?;
    let positional_inputs = &args[..positional_end];

    if let [input] = positional_inputs {
        validate_cell_text_input(input)?;
    }

    // Reject malformed call structure and host-only options before a resident
    // operand can trigger provider lookup or an upload/gather failure.
    if let Some(convert_from) = options.convert_from.as_deref() {
        if !convert_from.eq_ignore_ascii_case("datenum") {
            return Err(datetime_argument_error(format!(
                "datetime: unsupported ConvertFrom value '{convert_from}'"
            )));
        }
        if positional_inputs.len() != 1 {
            return Err(datetime_argument_error(
                "datetime: ConvertFrom='datenum' expects exactly one numeric input",
            ));
        }
    } else if !matches!(positional_inputs.len(), 0 | 1 | 3..=7) {
        return Err(datetime_argument_error(
            "datetime: unsupported argument pattern; use text, serial dates, or Y/M/D component inputs",
        ));
    }
    if options.input_format.is_some()
        && (options.convert_from.is_some()
            || !matches!(
                positional_inputs,
                [Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_)]
            ))
    {
        return Err(datetime_argument_error(
            "datetime: InputFormat requires exactly one text input",
        ));
    }
    if options.convert_from.is_none() && matches!(positional_inputs.len(), 4 | 5) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if positional_inputs
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if positional_inputs
        .iter()
        .any(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let positional = gather_args(positional_inputs).await?;

    if options.convert_from.is_some() {
        return numeric_value_to_datetime(positional[0].clone(), options.format);
    }

    match positional.len() {
        0 => {
            let now = Local::now().naive_local();
            datetime_object_from_serials(
                vec![datenum_from_naive(now)],
                vec![1, 1],
                options
                    .format
                    .unwrap_or_else(|| DEFAULT_DATETIME_FORMAT.to_string()),
            )
        }
        1 => match &positional[0] {
            Value::Object(obj) if obj.is_class(DATETIME_CLASS) => {
                let serials = serials_from_datetime_value(&positional[0])?;
                let format = options
                    .format
                    .unwrap_or_else(|| datetime_format_from_value(&positional[0]));
                datetime_object_from_serial_tensor(serials, format)
            }
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
                let (serials, shape, inferred_format) =
                    parse_text_input(positional[0].clone(), options.input_format.as_deref())?;
                datetime_object_from_serials(
                    serials,
                    shape,
                    options.format.unwrap_or(inferred_format),
                )
            }
            _ => {
                let numeric = tensor_from_numeric(positional[0].clone(), "date vector")?;
                if numeric.shape.len() == 2 && matches!(numeric.shape[1], 3 | 6) {
                    build_from_date_vectors(positional[0].clone(), options.format)
                } else {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &DATETIME_RAW_DATENUM_EXTENSION,
                        BUILTIN_NAME,
                    )?;
                    numeric_value_to_datetime(positional[0].clone(), options.format)
                }
            }
        },
        3 | 6 | 7 => build_from_components(positional, options.format),
        4 | 5 => build_from_components(positional, options.format),
        _ => unreachable!("constructor arity validated before resident access"),
    }
}
