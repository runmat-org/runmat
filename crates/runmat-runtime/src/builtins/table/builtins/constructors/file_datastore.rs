use super::*;
use runmat_builtins::{
    FileDatastoreOption, FileDatastoreReadMode, FILE_DATASTORE_BOOLEAN_EXTENSION,
    FILE_DATASTORE_GPU_INPUT_EXTENSION, FILE_DATASTORE_METADATA_EXTENSION,
};

fn extension(
    runtime: Option<&crate::context::RuntimeContext>,
    descriptor: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    match runtime {
        Some(runtime) => crate::compatibility::ensure_builtin_extension_enabled_in_context(
            runtime,
            descriptor,
            "fileDatastore",
        ),
        None => crate::compatibility::ensure_builtin_extension_enabled(descriptor, "fileDatastore"),
    }
}

/// Text locations/extensions are metadata, not evidence that a path exists.
fn public_text_spec(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) => true,
        Value::CharArray(value) => value.rows == 1 || value.data().is_empty(),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .all(|value| matches!(value, Value::CharArray(value) if value.rows == 1)),
        _ => false,
    }
}

fn include_subfolders(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(array) if array.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(0.0) => Ok(false),
        Value::Num(1.0) => Ok(true),
        Value::Tensor(tensor)
            if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F64 =>
        {
            let Some(runmat_value::NumericScalar::F64(number)) = tensor.numeric_value_at(0) else {
                unreachable!("F64 tensor owns an F64 scalar")
            };
            if number == 0.0 || number == 1.0 {
                return zero_one_bool_scalar(value, "IncludeSubfolders");
            }
            let flag = bool_scalar(&Value::Num(number), "IncludeSubfolders")?;
            extension(runtime, &FILE_DATASTORE_BOOLEAN_EXTENSION)?;
            Ok(flag)
        }
        _ => {
            // Validate first: extension mode does not admit malformed arrays,
            // non-finite values, or unrecognized text aliases.
            let flag = bool_scalar(value, "IncludeSubfolders")?;
            extension(runtime, &FILE_DATASTORE_BOOLEAN_EXTENSION)?;
            Ok(flag)
        }
    }
}

pub(super) async fn construct(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(invalid_argument(
            "fileDatastore: files location is required",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(invalid_argument(
            "fileDatastore: name-value options must be provided in pairs",
        ));
    }
    if args.iter().any(crate::value_contains_gpu) {
        extension(runtime, &FILE_DATASTORE_GPU_INPUT_EXTENSION)?;
    }
    let args = gather_retained_values(&args, runtime).await?;
    let files = args[0].clone();
    if !public_text_spec(&files) {
        extension(runtime, &FILE_DATASTORE_METADATA_EXTENSION)?;
    }
    let mut read_fcn = Value::string(String::new());
    let mut file_extensions = Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap());
    let mut include = false;
    let mut read_mode = Value::from("file");
    for index in (1..args.len()).step_by(2) {
        let name = scalar_text(&args[index], "fileDatastore option")?;
        let value = &args[index + 1];
        match FileDatastoreOption::parse(&name) {
            Some(FileDatastoreOption::ReadFcn) => read_fcn = value.clone(),
            Some(FileDatastoreOption::FileExtensions) => {
                let empty_character =
                    matches!(value, Value::CharArray(value) if value.data().is_empty());
                if !empty_character {
                    string_list(value).map_err(|_| {
                        invalid_argument(
                        "fileDatastore: FileExtensions must be text, a string array, or cellstr",
                    )
                    })?;
                }
                if !public_text_spec(value) {
                    extension(runtime, &FILE_DATASTORE_METADATA_EXTENSION)?;
                }
                file_extensions = value.clone();
            }
            Some(FileDatastoreOption::IncludeSubfolders) => {
                include = include_subfolders(runtime, value)?;
            }
            Some(FileDatastoreOption::ReadMode) => {
                let mode = scalar_text(value, "ReadMode")?;
                if FileDatastoreReadMode::parse(&mode).is_none() {
                    return Err(invalid_argument(
                        "fileDatastore: ReadMode must be 'file' or 'partialfile'",
                    ));
                }
                read_mode = Value::string(mode);
            }
            None => {
                return Err(invalid_argument(format!(
                    "fileDatastore: unsupported option '{name}'"
                )));
            }
        }
    }
    // Canonical callable identity classifies handles/closures without resolving
    // a name, selecting a target, inspecting captures, or invoking the reader.
    if read_fcn.callable_instance_key().is_none() {
        extension(runtime, &FILE_DATASTORE_METADATA_EXTENSION)?;
    }
    construct_descriptor_object(
        runtime,
        FILE_DATASTORE_CLASS,
        &[
            ("Files", files),
            ("ReadFcn", read_fcn),
            ("FileExtensions", file_extensions),
            ("IncludeSubfolders", Value::Bool(include)),
            ("ReadMode", read_mode),
        ],
    )
}
