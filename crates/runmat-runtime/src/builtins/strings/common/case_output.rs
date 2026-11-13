//! Exact retained UTF-8 output for the case-changing string-array builtins.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{
    exact_capacity_vec, ExactArrayInitializationError, ExactArrayStorage, SharedShape, StringArray,
    StringElement, Utf8Text,
};

const CHECKPOINT_CHUNK: usize = 4096;

mod character;
mod unicode16;
pub(crate) use character::transform_character_case;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextCase {
    Lower,
    Upper,
}

fn visit_case(
    input: &str,
    mode: TextCase,
    suffix_cased: Option<&[u8]>,
    mut visit: impl FnMut(char) -> Result<(), HostAllocationError>,
) -> Result<(), HostAllocationError> {
    let mut prefix_cased = false;
    for (byte_index, character) in input.char_indices() {
        match mode {
            TextCase::Lower => {
                if character == 'Σ'
                    && prefix_cased
                    && suffix_cased.is_some_and(|flags| flags[byte_index] == 0)
                {
                    visit('ς')?;
                } else {
                    for mapped in character.to_lowercase() {
                        visit(mapped)?;
                    }
                }
            }
            TextCase::Upper => {
                for mapped in character.to_uppercase() {
                    visit(mapped)?;
                }
            }
        }
        if !unicode16::is_case_ignorable(character) {
            prefix_cased = unicode16::is_cased(character);
        }
    }
    Ok(())
}

fn transformed_bytes(
    input: &str,
    mode: TextCase,
    runtime: Option<&RuntimeContext>,
) -> Result<usize, HostAllocationError> {
    let mut bytes = 0usize;
    let mut seen = 0usize;
    visit_case(input, mode, None, |character| {
        if seen.is_multiple_of(CHECKPOINT_CHUNK)
            && runtime.is_some_and(RuntimeContext::is_cancelled)
        {
            return Err(HostAllocationError::Cancelled);
        }
        bytes = bytes
            .checked_add(character.len_utf8())
            .ok_or(HostAllocationError::Overflow)?;
        seen = seen.checked_add(1).ok_or(HostAllocationError::Overflow)?;
        Ok(())
    })?;
    Ok(bytes)
}

fn prepare_sigma_suffix(
    input: &str,
    flags: &mut [u8],
    mut checkpoint: impl FnMut() -> Result<(), HostAllocationError>,
) -> Result<(), HostAllocationError> {
    let mut next_cased = false;
    for (index, (byte_index, character)) in input.char_indices().rev().enumerate() {
        if index % CHECKPOINT_CHUNK == 0 {
            checkpoint()?;
        }
        flags[byte_index] = u8::from(next_cased);
        if !unicode16::is_case_ignorable(character) {
            next_cased = unicode16::is_cased(character);
        }
    }
    Ok(())
}

fn make_exact_text(
    input: &str,
    mode: TextCase,
    bytes: usize,
    runtime: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    permit: &dyn runmat_value::ExactHostAllocationPermit,
    sigma_flags: &mut [u8],
) -> Result<Utf8Text, HostAllocationError> {
    let suffix_cased = if mode == TextCase::Lower && input.contains('Σ') {
        let flags = &mut sigma_flags[..input.len()];
        prepare_sigma_suffix(input, flags, || receipt.checkpoint(runtime))?;
        Some(&flags[..])
    } else {
        None
    };
    let mut output = exact_capacity_vec::<u8>(bytes).map_err(HostAllocationError::Numeric)?;
    let mut seen = 0usize;
    visit_case(input, mode, suffix_cased, |character| {
        if seen.is_multiple_of(CHECKPOINT_CHUNK) {
            receipt.checkpoint(runtime)?;
        }
        let mut encoded = [0u8; 4];
        let encoded = character.encode_utf8(&mut encoded).as_bytes();
        let next = output
            .len()
            .checked_add(encoded.len())
            .ok_or(HostAllocationError::Overflow)?;
        if next > bytes {
            return Err(HostAllocationError::Overflow);
        }
        output.extend_from_slice(encoded);
        seen = seen.checked_add(1).ok_or(HostAllocationError::Overflow)?;
        Ok(())
    })?;
    if output.len() != bytes {
        return Err(HostAllocationError::Overflow);
    }
    // SAFETY: every byte was appended from char::encode_utf8, and no other
    // write reaches this exact-capacity backing.
    let output = unsafe { String::from_utf8_unchecked(output) };
    Utf8Text::new(output)
        .try_admit_exact(permit)
        .map_err(|failure| HostAllocationError::Numeric(failure.source()))
}

pub(crate) fn transform_string_array_case(
    runtime: Option<&RuntimeContext>,
    input: StringArray,
    mode: TextCase,
    admission_error: impl Fn(&RuntimeContext, HostAllocationError) -> RuntimeError,
    invalid: impl Fn(String) -> RuntimeError,
) -> BuiltinResult<StringArray> {
    let ordinary = || -> BuiltinResult<StringArray> {
        let values = input
            .data()
            .iter()
            .map(|text| match text.as_str() {
                None => StringElement::Missing,
                Some(text) => StringElement::text(match mode {
                    TextCase::Lower => text.to_lowercase(),
                    TextCase::Upper => text.to_uppercase(),
                }),
            })
            .collect();
        StringArray::from_elements_unbounded(values, input.shape().to_vec())
            .map_err(|error| invalid(error))
    };
    let Some(runtime) = runtime else {
        return ordinary();
    };
    let mut physical = ExactArrayStorage::<StringElement>::requested_bytes(input.data().len())
        .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))?;
    let reuse_shape = input.shape.is_admitted();
    if !reuse_shape {
        physical = physical
            .checked_add(
                SharedShape::requested_bytes_for_rank(input.shape.len()).map_err(|cause| {
                    admission_error(runtime, HostAllocationError::Numeric(cause))
                })?,
            )
            .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
    }
    let mut claims = usize::from(!input.data().is_empty()) + usize::from(!reuse_shape);
    let mut sigma_scratch_len = 0usize;
    for text in input.data() {
        if runtime.is_cancelled() {
            return Err(admission_error(runtime, HostAllocationError::Cancelled));
        }
        let Some(text) = text.as_str() else { continue };
        if mode == TextCase::Lower && text.contains('Σ') {
            sigma_scratch_len = sigma_scratch_len.max(text.len());
        }
        let bytes = transformed_bytes(text, mode, Some(runtime))
            .map_err(|cause| admission_error(runtime, cause))?;
        physical = physical
            .checked_add(
                Utf8Text::requested_bytes_for_capacity(bytes).map_err(|cause| {
                    admission_error(runtime, HostAllocationError::Numeric(cause))
                })?,
            )
            .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
        claims = claims
            .checked_add(usize::from(bytes != 0))
            .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical)
            .map_err(|_| admission_error(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|cause| admission_error(runtime, cause))?;
    let scratch = if sigma_scratch_len == 0 {
        0
    } else {
        HostTypedWorkspace::<u8>::quoted_bytes(sigma_scratch_len)
            .map_err(|cause| admission_error(runtime, cause))?
    };
    let plan = HostAllocationPlan::new(scratch, retained)
        .map_err(|cause| admission_error(runtime, cause))?;
    let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|cause| admission_error(runtime, cause))?
    else {
        return ordinary();
    };
    let shape = if reuse_shape {
        input.shape.clone()
    } else {
        receipt
            .with_output_allocation(runtime, |permit| {
                SharedShape::new_admitted_from_slice(input.shape(), permit)
            })
            .map_err(|cause| admission_error(runtime, cause))?
    };
    let mut sigma_scratch = if sigma_scratch_len == 0 {
        None
    } else {
        Some(
            receipt
                .temporary_typed::<u8>(runtime, sigma_scratch_len)
                .map_err(|cause| admission_error(runtime, cause))?,
        )
    };
    let data = receipt
        .with_output_construction(runtime, |permit| {
            ExactArrayStorage::try_initialize(input.data().len(), permit, |index| {
                let text = &input.data()[index];
                receipt.checkpoint(runtime)?;
                let Some(text) = text.as_str() else {
                    return Ok(StringElement::Missing);
                };
                let bytes = transformed_bytes(text, mode, Some(runtime))?;
                let flags = sigma_scratch
                    .as_mut()
                    .map(|scratch| scratch.as_mut_slice())
                    .unwrap_or(&mut []);
                make_exact_text(text, mode, bytes, runtime, &receipt, permit, flags)
                    .map(StringElement::Text)
            })
        })
        .map_err(|cause| admission_error(runtime, cause))?
        .map_err(|cause| match cause {
            ExactArrayInitializationError::Allocation(cause) => {
                admission_error(runtime, HostAllocationError::Numeric(cause))
            }
            ExactArrayInitializationError::Element(cause) => admission_error(runtime, cause),
        })?;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| admission_error(runtime, cause))?;
    let output = StringArray::from_admitted(data, shape).map_err(|error| invalid(error))?;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| admission_error(runtime, cause))?;
    Ok(output)
}

pub(crate) fn transform_scalar_case(
    runtime: Option<&RuntimeContext>,
    input: StringElement,
    mode: TextCase,
    admission_error: impl Fn(&RuntimeContext, HostAllocationError) -> RuntimeError,
) -> BuiltinResult<StringElement> {
    if let Some(runtime) = runtime {
        if runtime.is_cancelled() {
            return Err(admission_error(runtime, HostAllocationError::Cancelled));
        }
    }
    let Some(input) = input.as_str() else {
        return Ok(StringElement::Missing);
    };
    let ordinary = || {
        let text = match mode {
            TextCase::Lower => input.to_lowercase(),
            TextCase::Upper => input.to_uppercase(),
        };
        Ok(StringElement::text(text))
    };
    let Some(runtime) = runtime else {
        return ordinary();
    };
    let bytes = transformed_bytes(input, mode, Some(runtime))
        .map_err(|cause| admission_error(runtime, cause))?;
    let physical = Utf8Text::requested_bytes_for_capacity(bytes)
        .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical)
            .map_err(|_| admission_error(runtime, HostAllocationError::Overflow))?,
        usize::from(bytes != 0),
    )
    .map_err(|cause| admission_error(runtime, cause))?;
    let sigma_scratch_len = if mode == TextCase::Lower && input.contains('Σ') {
        input.len()
    } else {
        0
    };
    let scratch = if sigma_scratch_len == 0 {
        0
    } else {
        HostTypedWorkspace::<u8>::quoted_bytes(sigma_scratch_len)
            .map_err(|cause| admission_error(runtime, cause))?
    };
    let plan = HostAllocationPlan::new(scratch, retained)
        .map_err(|cause| admission_error(runtime, cause))?;
    let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|cause| admission_error(runtime, cause))?
    else {
        return ordinary();
    };
    let mut sigma_scratch = if sigma_scratch_len == 0 {
        None
    } else {
        Some(
            receipt
                .temporary_typed::<u8>(runtime, sigma_scratch_len)
                .map_err(|cause| admission_error(runtime, cause))?,
        )
    };
    let output = receipt
        .with_output_construction(runtime, |permit| {
            let flags = sigma_scratch
                .as_mut()
                .map(|scratch| scratch.as_mut_slice())
                .unwrap_or(&mut []);
            make_exact_text(input, mode, bytes, runtime, &receipt, permit, flags)
        })
        .map_err(|cause| admission_error(runtime, cause))?
        .map_err(|cause| admission_error(runtime, cause))?;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| admission_error(runtime, cause))?;
    Ok(StringElement::Text(output))
}

#[cfg(test)]
#[path = "case_output/tests.rs"]
mod tests;
