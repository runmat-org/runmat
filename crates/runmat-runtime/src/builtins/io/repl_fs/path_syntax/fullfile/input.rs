use runmat_value::Value;

use super::super::text::TextContainer;
use crate::builtins::io::repl_fs::text_conversion::tensor_char_codes_to_string;
use crate::BuiltinResult;

const IDENTITY: &str = "fullfile";

pub(super) async fn decode(value: &Value) -> BuiltinResult<TextContainer> {
    match value {
        Value::Tensor(_) | Value::GpuTensor(_) => decode_numeric(value).await,
        _ => TextContainer::decode(
            value,
            IDENTITY,
            &runmat_builtins::FULLFILE_ERROR_ARGUMENT_TYPE,
        ),
    }
}

async fn decode_numeric(value: &Value) -> BuiltinResult<TextContainer> {
    crate::compatibility::ensure_builtin_extension_enabled(
        &runmat_builtins::FULLFILE_NUMERIC_CHARACTER_CODES_EXTENSION,
        IDENTITY,
    )?;
    let gathered = crate::gather_if_needed_async(value)
        .await
        .map_err(|source| {
            super::super::error::source(&runmat_builtins::FULLFILE_ERROR_PROVIDER, IDENTITY, source)
        })?;
    let Value::Tensor(tensor) = gathered else {
        return Err(super::super::error::catalog(
            &runmat_builtins::FULLFILE_ERROR_ARGUMENT_TYPE,
            IDENTITY,
        ));
    };
    if tensor.rows() != 1 || tensor.shape.len() > 2 {
        return Err(super::super::error::catalog(
            &runmat_builtins::FULLFILE_ERROR_ARGUMENT_TYPE,
            IDENTITY,
        ));
    }
    let text = tensor_char_codes_to_string(&tensor).ok_or_else(|| {
        super::super::error::catalog(&runmat_builtins::FULLFILE_ERROR_ARGUMENT_TYPE, IDENTITY)
    })?;
    Ok(TextContainer {
        representation: super::super::text::TextRepresentation::Character,
        values: vec![text],
        shape: vec![1, 1],
    })
}
