//! Borrow existing scalar text or admit only a newly encoded character row.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::RuntimeError;
use runmat_value::{Utf16TextCopyError, Utf8Text, Value};

pub(crate) fn text_input_admission(
    name: &'static str,
    runtime: &RuntimeContext,
    cause: HostAllocationError,
) -> RuntimeError {
    crate::build_runtime_error(format!("{name}: host allocation: {cause}"))
        .with_builtin(name)
        .with_identifier(format!(
            "{}:{}",
            runtime.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

pub(crate) enum TextInput<'a> {
    Borrowed(&'a str),
    Owned(Utf8Text),
}

impl TextInput<'_> {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            Self::Borrowed(text) => text,
            Self::Owned(text) => text.as_str(),
        }
    }
}

pub(crate) fn scalar_text_input<'a>(
    value: &'a Value,
    runtime: Option<&RuntimeContext>,
    admission_error: impl Fn(&RuntimeContext, HostAllocationError) -> RuntimeError,
) -> Result<Option<TextInput<'a>>, RuntimeError> {
    match value {
        Value::String(text) => Ok(text.as_str().map(TextInput::Borrowed)),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(array.data()[0].as_str().map(TextInput::Borrowed))
        }
        Value::CharArray(array) if array.rows == 0 => Ok(Some(TextInput::Borrowed(""))),
        Value::CharArray(array) if array.rows == 1 => {
            let chars = array.data().get(..array.cols).ok_or_else(|| {
                crate::build_runtime_error("text input: character shape exceeds its payload")
                    .build()
            })?;
            let Some(runtime) = runtime else {
                let text = runmat_value::decode_utf16_code_units(chars).map_err(invalid_utf16)?;
                return Ok(Some(TextInput::Owned(Utf8Text::new(text))));
            };
            let (bytes, physical) = Utf8Text::requested_bytes_for_utf16_with(chars, |_| {
                if runtime.is_cancelled() {
                    Err(HostAllocationError::Cancelled)
                } else {
                    Ok(())
                }
            })
            .map_err(|source| utf16_copy_error(runtime, &admission_error, source))?;
            let retained = HostAllocationPlan::retained_exact_output_bytes(
                u64::try_from(physical)
                    .map_err(|_| admission_error(runtime, HostAllocationError::Overflow))?,
                usize::from(bytes != 0),
            )
            .map_err(|cause| admission_error(runtime, cause))?;
            let plan = HostAllocationPlan::new(0, retained)
                .map_err(|cause| admission_error(runtime, cause))?;
            let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan)
                .map_err(|cause| admission_error(runtime, cause))?
            else {
                let text = runmat_value::decode_utf16_code_units(chars).map_err(invalid_utf16)?;
                return Ok(Some(TextInput::Owned(Utf8Text::new(text))));
            };
            let permit = receipt
                .output_permit(runtime)
                .map_err(|cause| admission_error(runtime, cause))?;
            let encoded = Utf8Text::try_copy_utf16_exact_with(chars, &permit, |_| {
                receipt.checkpoint(runtime)
            })
            .map_err(|source| utf16_copy_error(runtime, &admission_error, source))?;
            receipt
                .checkpoint(runtime)
                .map_err(|cause| admission_error(runtime, cause))?;
            Ok(Some(TextInput::Owned(encoded)))
        }
        _ => Ok(None),
    }
}

fn invalid_utf16(source: runmat_value::Utf16DecodeError) -> RuntimeError {
    crate::build_runtime_error("text input contains malformed UTF-16")
        .with_source(source)
        .build()
}

fn utf16_copy_error(
    runtime: &RuntimeContext,
    admission_error: &impl Fn(&RuntimeContext, HostAllocationError) -> RuntimeError,
    source: runmat_value::ExactArrayInitializationError<Utf16TextCopyError<HostAllocationError>>,
) -> RuntimeError {
    match source {
        runmat_value::ExactArrayInitializationError::Allocation(source) => {
            admission_error(runtime, HostAllocationError::Numeric(source))
        }
        runmat_value::ExactArrayInitializationError::Element(Utf16TextCopyError::Checkpoint(
            source,
        )) => admission_error(runtime, source),
        runmat_value::ExactArrayInitializationError::Element(Utf16TextCopyError::Decode(
            source,
        )) => invalid_utf16(source),
    }
}
