use runmat_value::{Tensor, Value};

use crate::builtins::io::repl_fs::text_conversion::tensor_char_codes_to_string;

pub(super) struct Request {
    pub(super) root: Option<String>,
    pub(super) excludes: Option<String>,
}

pub(super) async fn decode(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Request> {
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
        Value::String(runmat_value::StringElement::Text(_)) => Ok(()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(())
        }
        Value::CharArray(chars) if is_row(&chars.shape) => Ok(()),
        Value::Tensor(tensor) if is_row(tensor.shape()) => require_numeric_extension(),
        Value::GpuTensor(handle) if is_row(handle.shape()) => require_numeric_extension(),
        _ => Err(super::errors::descriptor(type_error)),
    }
}

async fn text(
    value: Value,
    type_error: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> crate::BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.into_string_unbounded()),
        Value::StringArray(array) => Ok(array.data()[0]
            .as_str()
            .ok_or_else(|| super::errors::descriptor(type_error))?
            .to_owned()),
        Value::CharArray(chars) => {
            crate::builtins::io::utf16::decode_utf16_with_error(chars.data(), || {
                super::errors::descriptor(type_error)
            })
        }
        Value::Tensor(tensor) => numeric_text(&tensor, type_error),
        value @ Value::GpuTensor(_) => {
            let gathered = crate::gather_if_needed_async(&value)
                .await
                .map_err(|error| {
                    let mut context = super::errors::detail(
                        &runmat_builtins::GENPATH_ERROR_PROVIDER,
                        error.message(),
                    );
                    context.source = Some(Box::new(error));
                    context
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
    tensor_char_codes_to_string(tensor).ok_or_else(|| super::errors::descriptor(type_error))
}

fn is_row(shape: &[usize]) -> bool {
    crate::builtins::common::shape::effective_rank(shape) <= 2
        && shape.first().copied().unwrap_or(1) == 1
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
