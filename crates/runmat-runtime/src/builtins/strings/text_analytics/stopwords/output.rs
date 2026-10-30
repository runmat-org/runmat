//! Exact output ownership for the fixed stop-word vocabulary.

use super::stopwords_error;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{
    exact_capacity_vec, ExactArrayInitializationError, ExactArrayStorage, SharedShape, StringArray,
    StringElement, Utf8Text,
};

fn admission_error(runtime: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    crate::builtins::strings::common::text_input_admission("stopWords", runtime, cause)
}

pub(super) fn string_array(
    runtime: Option<&RuntimeContext>,
    words: &[&str],
) -> BuiltinResult<StringArray> {
    let ordinary = || {
        StringArray::new(
            words.iter().map(|word| (*word).to_owned()).collect(),
            vec![words.len(), 1],
        )
        .map_err(|error| stopwords_error(format!("stopWords: {error}")))
    };
    let Some(runtime) = runtime else {
        return ordinary();
    };
    let mut physical = ExactArrayStorage::<StringElement>::requested_bytes(words.len())
        .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))?
        .checked_add(
            SharedShape::requested_bytes_for_rank(2)
                .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))?,
        )
        .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
    let mut claims = usize::from(!words.is_empty()) + 1;
    for (index, word) in words.iter().enumerate() {
        if index % 4096 == 0 && runtime.is_cancelled() {
            return Err(admission_error(runtime, HostAllocationError::Cancelled));
        }
        physical =
            physical
                .checked_add(Utf8Text::requested_bytes_for_capacity(word.len()).map_err(
                    |cause| admission_error(runtime, HostAllocationError::Numeric(cause)),
                )?)
                .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
        claims = claims
            .checked_add(usize::from(!word.is_empty()))
            .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical)
            .map_err(|_| admission_error(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|cause| admission_error(runtime, cause))?;
    let plan =
        HostAllocationPlan::new(0, retained).map_err(|cause| admission_error(runtime, cause))?;
    let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|cause| admission_error(runtime, cause))?
    else {
        return ordinary();
    };
    let shape = receipt
        .with_output_allocation(runtime, |permit| {
            SharedShape::new_admitted_from_slice(&[words.len(), 1], permit)
        })
        .map_err(|cause| admission_error(runtime, cause))?;
    let data = receipt
        .with_output_construction(runtime, |permit| {
            ExactArrayStorage::try_initialize(words.len(), permit, |index| {
                if index % 4096 == 0 {
                    receipt.checkpoint(runtime)?;
                }
                let source = words[index];
                let mut bytes =
                    exact_capacity_vec::<u8>(source.len()).map_err(HostAllocationError::Numeric)?;
                bytes.extend_from_slice(source.as_bytes());
                // SAFETY: the source was valid UTF-8 and is copied unchanged.
                Utf8Text::new(unsafe { String::from_utf8_unchecked(bytes) })
                    .try_admit_exact(permit)
                    .map(StringElement::Text)
                    .map_err(|failure| HostAllocationError::Numeric(failure.source()))
            })
        })
        .map_err(|cause| admission_error(runtime, cause))?
        .map_err(|cause| match cause {
            ExactArrayInitializationError::Allocation(cause) => {
                admission_error(runtime, HostAllocationError::Numeric(cause))
            }
            ExactArrayInitializationError::Element(cause) => admission_error(runtime, cause),
        })?;
    let output = StringArray::from_admitted(data, shape)
        .map_err(|error| stopwords_error(format!("stopWords: {error}")))?;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| admission_error(runtime, cause))?;
    Ok(output)
}
