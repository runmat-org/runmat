use super::capabilities::{
    BUILTIN_NAME, DATETIME_CLASS, DATETIME_GPU_INPUT_EXTENSION,
    DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION, DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_RAW_DATENUM_EXTENSION, DEFAULT_DATETIME_FORMAT,
};
use super::*;

#[runmat_macros::runtime_builtin(
    name = "datetime",
    descriptor(crate::builtins::datetime::DATETIME_DESCRIPTOR),
    extensions(crate::builtins::datetime::DATETIME_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::DATETIME_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create datetime arrays from text, components, or serial date numbers.",
    keywords = "datetime,date,time,datenum,Format",
    related = "year,month,day,hour,minute,second,string,char,disp",
    examples = "t = datetime(2024, 4, 9, 13, 30, 0);"
)]
pub(super) async fn datetime_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_datetime_class_registered();
    if args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args
        .iter()
        .any(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let args = gather_args(&args).await?;
    let (positional_end, options) = parse_trailing_options(&args)?;
    let positional = args[..positional_end].to_vec();

    if let Some(convert_from) = options.convert_from {
        if !convert_from.eq_ignore_ascii_case("datenum") {
            return Err(datetime_error(format!(
                "datetime: unsupported ConvertFrom value '{convert_from}'"
            )));
        }
        if positional.len() != 1 {
            return Err(datetime_error(
                "datetime: ConvertFrom='datenum' expects exactly one numeric input",
            ));
        }
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
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
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
        4 | 5 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION,
                BUILTIN_NAME,
            )?;
            build_from_components(positional, options.format)
        }
        _ => Err(datetime_error(
            "datetime: unsupported argument pattern; use text, serial dates, or Y/M/D component inputs",
        )),
    }
}
