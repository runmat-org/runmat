use runmat_value::{Tensor, Value};

use crate::{runtime_descriptor_error, runtime_descriptor_error_with_detail, BuiltinResult};

const BUILTIN_NAME: &str = "path";

pub(super) async fn text(value: Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::Tensor(tensor) => numeric_text(tensor),
        Value::GpuTensor(_) => resident_numeric_text(value).await,
        _ => Err(invalid_input()),
    }
}

fn numeric_text(tensor: Tensor) -> BuiltinResult<String> {
    require_numeric_extension()?;
    decode_numeric_row(&tensor)
}

async fn resident_numeric_text(value: Value) -> BuiltinResult<String> {
    require_numeric_extension()?;
    let gathered = crate::gather_if_needed_async(&value)
        .await
        .map_err(|error| {
            runtime_descriptor_error_with_detail(
                BUILTIN_NAME,
                &runmat_builtins::PATH_ERROR_PROVIDER_FAILED,
                error.message(),
            )
        })?;
    match gathered {
        Value::Tensor(tensor) => decode_numeric_row(&tensor),
        _ => Err(invalid_input()),
    }
}

fn require_numeric_extension() -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &runmat_builtins::PATH_NUMERIC_CHARACTER_CODES_EXTENSION,
        BUILTIN_NAME,
    )
}

fn decode_numeric_row(tensor: &Tensor) -> BuiltinResult<String> {
    if tensor.shape.len() > 2 || tensor.rows() > 1 {
        return Err(invalid_input());
    }
    super::super::tensor_char_codes_to_string(tensor).ok_or_else(invalid_input)
}

fn invalid_input() -> crate::RuntimeError {
    runtime_descriptor_error(BUILTIN_NAME, &runmat_builtins::PATH_ERROR_INVALID_INPUT)
}
