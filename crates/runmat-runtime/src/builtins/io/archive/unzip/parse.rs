use super::*;

pub(super) struct UnzipRequest {
    pub(super) archive: runmat_builtins::UnzipSourceRoute,
    pub(super) output_folder: Option<String>,
}

pub(super) fn parse_arguments(args: &[Value]) -> BuiltinResult<UnzipRequest> {
    match args {
        [archive] => Ok(UnzipRequest {
            archive: runmat_builtins::UnzipSourceRoute::parse(value_to_string_scalar(
                archive,
                &UNZIP_ERROR_FILENAME,
            )?),
            output_folder: None,
        }),
        [archive, output_folder] => Ok(UnzipRequest {
            archive: runmat_builtins::UnzipSourceRoute::parse(value_to_string_scalar(
                archive,
                &UNZIP_ERROR_FILENAME,
            )?),
            output_folder: Some(value_to_string_scalar(
                output_folder,
                &UNZIP_ERROR_OUTPUT_FOLDER,
            )?),
        }),
        [] => Err(unzip_error_with(
            &UNZIP_ERROR_INVALID_ARGUMENT,
            "unzip: ZIP filename is required",
        )),
        _ => Err(unzip_error_with(
            &UNZIP_ERROR_INVALID_ARGUMENT,
            "unzip: expected ZIP filename and optional output folder",
        )),
    }
}

fn value_to_string_scalar(
    value: &Value,
    descriptor: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(strings) if strings.data.len() == 1 => Ok(strings.data[0].clone()),
        _ => Err(unzip_error_with(
            descriptor,
            "unzip: expected a string scalar or character vector",
        )),
    }
    .and_then(|text| {
        if text.trim().is_empty() {
            Err(unzip_error_with(
                descriptor,
                "unzip: path arguments must not be empty",
            ))
        } else {
            Ok(text)
        }
    })
}
