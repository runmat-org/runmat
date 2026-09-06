use runmat_value::{Tensor, Value};

pub(super) struct Request {
    pub(super) root: Option<String>,
    pub(super) excludes: Option<String>,
}

pub(super) async fn decode(args: Vec<Value>) -> crate::BuiltinResult<Request> {
    if args.len() > 2 {
        return Err(super::errors::descriptor(
            &runmat_builtins::GENPATH_ERROR_TOO_MANY_INPUTS,
        ));
    }
    if args.len() == 2 {
        require_extension(&runmat_builtins::GENPATH_EXCLUDES_EXTENSION)?;
    }
    for (index, value) in args.iter().enumerate() {
        preflight(value, error_for(index))?;
    }

    let mut values = Vec::with_capacity(args.len());
    for (index, value) in args.into_iter().enumerate() {
        values.push(text(value, error_for(index)).await?);
    }
    Ok(Request {
        root: values.first().cloned(),
        excludes: values.get(1).cloned(),
    })
}

fn preflight(
    value: &Value,
    type_error: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> crate::BuiltinResult<()> {
    match value {
        Value::String(_) => Ok(()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(()),
        Value::CharArray(chars) if chars.shape.len() <= 2 && chars.rows == 1 => Ok(()),
        Value::Tensor(tensor) if is_row(&tensor.shape) => require_numeric_extension(),
        Value::GpuTensor(handle) if is_row(&handle.shape) => require_numeric_extension(),
        _ => Err(super::errors::descriptor(type_error)),
    }
}

async fn text(
    value: Value,
    type_error: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> crate::BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text),
        Value::StringArray(array) => Ok(array.data[0].clone()),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
        Value::Tensor(tensor) => numeric_text(&tensor, type_error),
        value @ Value::GpuTensor(_) => {
            let gathered = crate::gather_if_needed_async(&value)
                .await
                .map_err(|error| {
                    super::errors::detail(&runmat_builtins::GENPATH_ERROR_PROVIDER, error.message())
                })?;
            let Value::Tensor(tensor) = gathered else {
                return Err(super::errors::descriptor(type_error));
            };
            numeric_text(&tensor, type_error)
        }
        _ => Err(super::errors::descriptor(type_error)),
    }
}

fn numeric_text(
    tensor: &Tensor,
    type_error: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> crate::BuiltinResult<String> {
    super::super::tensor_char_codes_to_string(tensor)
        .ok_or_else(|| super::errors::descriptor(type_error))
}

fn is_row(shape: &[usize]) -> bool {
    shape.len() <= 2 && shape.first().copied().unwrap_or(1) == 1
}

fn require_numeric_extension() -> crate::BuiltinResult<()> {
    require_extension(&runmat_builtins::GENPATH_NUMERIC_CHARACTER_CODES_EXTENSION)
}

fn require_extension(
    extension: &'static runmat_builtins::BuiltinExtensionDescriptor,
) -> crate::BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, super::errors::NAME)
}

fn error_for(index: usize) -> &'static runmat_builtins::BuiltinErrorDescriptor {
    if index == 0 {
        &runmat_builtins::GENPATH_ERROR_FOLDER_TYPE
    } else {
        &runmat_builtins::GENPATH_ERROR_EXCLUDES_TYPE
    }
}
