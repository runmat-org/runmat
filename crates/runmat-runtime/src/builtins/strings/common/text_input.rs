//! Borrow existing scalar text or admit only a newly encoded character row.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::RuntimeError;
use runmat_value::{exact_capacity_vec, Utf8Text, Value};

const CHECKPOINT_CHUNK: usize = 4096;

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
        Value::String(text) => Ok(Some(TextInput::Borrowed(text.as_str()))),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(Some(TextInput::Borrowed(array.data()[0].as_str())))
        }
        Value::CharArray(array) if array.rows == 0 => Ok(Some(TextInput::Borrowed(""))),
        Value::CharArray(array) if array.rows == 1 => {
            let chars = array.data().get(..array.cols).ok_or_else(|| {
                crate::build_runtime_error("text input: character shape exceeds its payload")
                    .build()
            })?;
            let ordinary = || {
                let text: String = chars.iter().collect();
                TextInput::Owned(Utf8Text::new(text))
            };
            let Some(runtime) = runtime else {
                return Ok(Some(ordinary()));
            };
            let mut bytes = 0usize;
            for (index, character) in chars.iter().enumerate() {
                if index % CHECKPOINT_CHUNK == 0 && runtime.is_cancelled() {
                    return Err(admission_error(runtime, HostAllocationError::Cancelled));
                }
                bytes = bytes
                    .checked_add(character.len_utf8())
                    .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
            }
            let physical = Utf8Text::requested_bytes_for_capacity(bytes)
                .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))?;
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
                return Ok(Some(ordinary()));
            };
            let encoded = receipt
                .with_output_construction(runtime, |permit| {
                    let mut output =
                        exact_capacity_vec::<u8>(bytes).map_err(HostAllocationError::Numeric)?;
                    for (index, character) in chars.iter().enumerate() {
                        if index % CHECKPOINT_CHUNK == 0 {
                            receipt.checkpoint(runtime)?;
                        }
                        let mut scratch = [0u8; 4];
                        output.extend_from_slice(character.encode_utf8(&mut scratch).as_bytes());
                    }
                    if output.len() != bytes {
                        return Err(HostAllocationError::Overflow);
                    }
                    // SAFETY: only char::encode_utf8 wrote this byte backing.
                    Utf8Text::new(unsafe { String::from_utf8_unchecked(output) })
                        .try_admit_exact(permit)
                        .map_err(|failure| HostAllocationError::Numeric(failure.source()))
                })
                .map_err(|cause| admission_error(runtime, cause))?
                .map_err(|cause| admission_error(runtime, cause))?;
            receipt
                .checkpoint(runtime)
                .map_err(|cause| admission_error(runtime, cause))?;
            Ok(Some(TextInput::Owned(encoded)))
        }
        _ => Ok(None),
    }
}
