use runmat_value::{Tensor, Value};

use crate::builtins::io::repl_fs::text_conversion::tensor_char_codes_to_string;

pub(super) struct Request {
    pub(super) filename: Option<String>,
    pub(super) requested_outputs: Option<usize>,
}

pub(super) async fn decode(args: Vec<Value>) -> crate::BuiltinResult<Request> {
    if args.len() > 1 {
        return Err(super::errors::descriptor(
            &runmat_builtins::SAVEPATH_ERROR_TOO_MANY_INPUTS,
        ));
    }
    let requested_outputs = crate::output_count::current_output_count();
    let output_count = requested_outputs.unwrap_or(1);
    if output_count > 3 {
        return Err(super::errors::descriptor(
            &runmat_builtins::SAVEPATH_ERROR_TOO_MANY_OUTPUTS,
        ));
    }
    if output_count > 1 {
        require_extension(&runmat_builtins::SAVEPATH_DIAGNOSTIC_OUTPUTS_EXTENSION)?;
    }
    if let Some(value) = args.first() {
        preflight(value)?;
    }
    let filename = match args.into_iter().next() {
        Some(value) => {
            let text = decode_text(value).await?;
            if text.is_empty() {
                return Err(super::errors::descriptor(
                    &runmat_builtins::SAVEPATH_ERROR_EMPTY_FILENAME,
                ));
            }
            Some(text)
        }
        None => None,
    };
    Ok(Request {
        filename,
        requested_outputs,
    })
}

fn preflight(value: &Value) -> crate::BuiltinResult<()> {
    match value {
        Value::String(_) => Ok(()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(()),
        Value::CharArray(chars) if chars.shape.len() <= 2 && chars.rows == 1 => Ok(()),
        Value::Tensor(tensor) if is_row(&tensor.shape) => require_numeric_extension(),
        Value::GpuTensor(handle) if is_row(&handle.shape) => require_numeric_extension(),
        _ => Err(super::errors::descriptor(
            &runmat_builtins::SAVEPATH_ERROR_ARGUMENT_TYPE,
        )),
    }
}

async fn decode_text(value: Value) -> crate::BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text),
        Value::StringArray(array) => Ok(array.data[0].clone()),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
        Value::Tensor(tensor) => numeric_text(&tensor),
        value @ Value::GpuTensor(_) => {
            let gathered = crate::gather_if_needed_async(&value)
                .await
                .map_err(|error| {
                    super::errors::detail(
                        &runmat_builtins::SAVEPATH_ERROR_PROVIDER,
                        error.message(),
                    )
                })?;
            let Value::Tensor(tensor) = gathered else {
                return Err(type_error());
            };
            numeric_text(&tensor)
        }
        _ => Err(type_error()),
    }
}

fn numeric_text(tensor: &Tensor) -> crate::BuiltinResult<String> {
    tensor_char_codes_to_string(tensor).ok_or_else(type_error)
}

fn is_row(shape: &[usize]) -> bool {
    shape.len() <= 2 && shape.first().copied().unwrap_or(1) == 1
}

fn require_numeric_extension() -> crate::BuiltinResult<()> {
    require_extension(&runmat_builtins::SAVEPATH_NUMERIC_CHARACTER_CODES_EXTENSION)
}

fn require_extension(
    extension: &'static runmat_builtins::BuiltinExtensionDescriptor,
) -> crate::BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, super::errors::NAME)
}

fn type_error() -> crate::RuntimeError {
    super::errors::descriptor(&runmat_builtins::SAVEPATH_ERROR_ARGUMENT_TYPE)
}
