use runmat_value::{Tensor, Value};

use crate::builtins::io::repl_fs::text_conversion::tensor_char_codes_to_string;
use crate::{runtime_descriptor_error, runtime_descriptor_error_with_detail, BuiltinResult};

const BUILTIN_NAME: &str = "path";

pub(super) enum AdmittedText {
    Text(String),
    Resident(Value),
}

pub(super) fn admit(value: Value) -> BuiltinResult<AdmittedText> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(AdmittedText::Text(text.into_string_unbounded()))
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(AdmittedText::Text(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        Value::CharArray(chars)
            if chars.rows == 1
                && crate::builtins::common::shape::effective_rank(&chars.shape) <= 2 =>
        {
            crate::builtins::common::text::decode_utf16_with_error(chars.data(), invalid_input)
                .map(AdmittedText::Text)
        }
        Value::Tensor(tensor) => numeric_text(tensor).map(AdmittedText::Text),
        Value::GpuTensor(ref handle) if numeric_row(handle.shape()) => {
            require_numeric_extension()?;
            Ok(AdmittedText::Resident(value))
        }
        _ => Err(invalid_input()),
    }
}

impl AdmittedText {
    pub(super) async fn resolve(self) -> BuiltinResult<String> {
        match self {
            Self::Text(text) => Ok(text),
            Self::Resident(value) => resident_numeric_text(value).await,
        }
    }
}

fn numeric_text(tensor: Tensor) -> BuiltinResult<String> {
    require_numeric_extension()?;
    decode_numeric_row(&tensor)
}

async fn resident_numeric_text(value: Value) -> BuiltinResult<String> {
    let gathered = crate::gather_if_needed_async(&value)
        .await
        .map_err(|error| {
            let mut context = runtime_descriptor_error_with_detail(
                BUILTIN_NAME,
                &runmat_builtins::PATH_ERROR_PROVIDER_FAILED,
                error.message(),
            );
            context.source = Some(Box::new(error));
            context
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
    if !numeric_row(tensor.shape()) {
        return Err(invalid_input());
    }
    tensor_char_codes_to_string(tensor).ok_or_else(invalid_input)
}

fn numeric_row(shape: &[usize]) -> bool {
    crate::builtins::common::shape::effective_rank(shape) <= 2
        && shape.first().copied().unwrap_or(1) <= 1
}

fn invalid_input() -> crate::RuntimeError {
    runtime_descriptor_error(BUILTIN_NAME, &runmat_builtins::PATH_ERROR_INVALID_INPUT)
}
