//! Retained string-array replacement for the missing-value operations.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{
    ExactArrayInitializationError, ExactArrayStorage, SharedShape, StringArray, Utf8Text,
};

mod outputs;
mod literal;
pub(super) use literal::TextLiteral;
pub(super) use outputs::fill_outputs;

const CHECKPOINT_CHUNK: usize = 4096;

#[derive(Clone, Copy)]
pub(super) enum TextSelection {
    Input(usize),
    Literal,
}

fn initialization_error(
    error: ExactArrayInitializationError<HostAllocationError>,
) -> HostAllocationError {
    match error {
        ExactArrayInitializationError::Allocation(cause) => HostAllocationError::Numeric(cause),
        ExactArrayInitializationError::Element(cause) => cause,
    }
}

pub(super) fn selected_string_array(
    runtime: Option<&RuntimeContext>,
    input: StringArray,
    literal: TextLiteral<'_>,
    select: impl Fn(&StringArray, usize) -> TextSelection,
    admission_error: impl Fn(&RuntimeContext, HostAllocationError) -> RuntimeError,
    invalid: impl Fn(String) -> RuntimeError,
) -> BuiltinResult<StringArray> {
    // A selector may name only an original non-replaced source, its own slot,
    // or the literal. The unique-owner path mutates in place; Fill's source
    // map is built from original nonmissing line boundaries before entry.
    let Some(runtime) = runtime else {
        return ordinary(input, literal, select, invalid);
    };
    if HostAllocationReceipt::policy(Some(runtime))
        .map_err(|cause| admission_error(runtime, cause))?
        == HostAllocationPolicy::Unbounded
    {
        return ordinary(input, literal, select, invalid);
    }

    let count = input.data.len();
    let reuse_data = input.data.is_exact() && input.data.is_unique();
    let reuse_shape = input.shape.is_admitted();
    let mut physical = 0usize;
    let mut claims = 0usize;
    if !reuse_data {
        physical = ExactArrayStorage::<Utf8Text>::requested_bytes(count)
            .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))?;
        claims += usize::from(count != 0);
    }
    if !reuse_shape {
        physical = physical
            .checked_add(
                SharedShape::requested_bytes_for_rank(input.shape.len()).map_err(|cause| {
                    admission_error(runtime, HostAllocationError::Numeric(cause))
                })?,
            )
            .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
        claims += 1;
    }
    let mut uses_literal = false;
    for index in 0..count {
        if index % CHECKPOINT_CHUNK == 0 && runtime.is_cancelled() {
            return Err(admission_error(runtime, HostAllocationError::Cancelled));
        }
        match select(&input, index) {
            TextSelection::Literal => uses_literal = true,
            TextSelection::Input(source) => {
                let text = input
                    .data
                    .get(source)
                    .ok_or_else(|| invalid("text source index exceeds input".into()))?;
                if !text.has_retained_claim() {
                    physical = physical
                        .checked_add(
                            Utf8Text::requested_bytes_for_capacity(text.as_str().len()).map_err(
                                |cause| {
                                    admission_error(runtime, HostAllocationError::Numeric(cause))
                                },
                            )?,
                        )
                        .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
                    claims = claims
                        .checked_add(usize::from(!text.as_str().is_empty()))
                        .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
                }
            }
        }
    }
    if uses_literal {
        physical = physical
            .checked_add(
                literal
                    .requested_bytes(runtime)
                    .map_err(|cause| admission_error(runtime, cause))?,
            )
            .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
        claims = claims
            .checked_add(usize::from(!literal.is_empty()))
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
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|cause| admission_error(runtime, cause))?
        .ok_or_else(|| admission_error(runtime, HostAllocationError::MissingManagedOwner))?;
    let shape = if reuse_shape {
        input.shape.clone()
    } else {
        receipt
            .with_output_allocation(runtime, |permit| {
                SharedShape::new_admitted_from_slice(&input.shape, permit)
            })
            .map_err(|cause| admission_error(runtime, cause))?
    };
    let literal_owner = if uses_literal {
        Some(
            receipt
                .with_output_construction(runtime, |permit| {
                    literal.copy_exact(permit, &receipt, runtime)
                })
                .map_err(|cause| admission_error(runtime, cause))?
                .map_err(|cause| admission_error(runtime, initialization_error(cause)))?,
        )
    } else {
        None
    };
    if reuse_data {
        let mut output = input;
        for index in 0..count {
            if index % CHECKPOINT_CHUNK == 0 {
                receipt
                    .checkpoint(runtime)
                    .map_err(|cause| admission_error(runtime, cause))?;
            }
            let selected = select(&output, index);
            let replacement =
                selected_text(&output, selected, literal_owner.as_ref(), &receipt, runtime)
                    .map_err(|cause| admission_error(runtime, cause))?;
            output
                .data
                .try_as_mut_slice_unique()
                .ok_or_else(|| admission_error(runtime, HostAllocationError::StaleAuthority))?
                [index] = replacement;
        }
        output.shape = shape;
        let result = StringArray::from_admitted(output.data, output.shape).map_err(invalid)?;
        receipt
            .checkpoint(runtime)
            .map_err(|cause| admission_error(runtime, cause))?;
        return Ok(result);
    }
    let data = receipt
        .with_output_construction(runtime, |permit| {
            ExactArrayStorage::try_initialize(count, permit, |index| {
                if index % CHECKPOINT_CHUNK == 0 {
                    receipt.checkpoint(runtime)?;
                }
                let selected = select(&input, index);
                selected_text(&input, selected, literal_owner.as_ref(), &receipt, runtime)
            })
        })
        .map_err(|cause| admission_error(runtime, cause))?
        .map_err(|cause| match cause {
            runmat_value::ExactArrayInitializationError::Allocation(cause) => {
                admission_error(runtime, HostAllocationError::Numeric(cause))
            }
            runmat_value::ExactArrayInitializationError::Element(cause) => {
                admission_error(runtime, cause)
            }
        })?;
    let result = StringArray::from_admitted(data, shape).map_err(invalid)?;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| admission_error(runtime, cause))?;
    Ok(result)
}

fn selected_text(
    input: &StringArray,
    selection: TextSelection,
    literal: Option<&Utf8Text>,
    receipt: &HostAllocationReceipt,
    runtime: &RuntimeContext,
) -> Result<Utf8Text, HostAllocationError> {
    match selection {
        TextSelection::Literal => Ok(literal.expect("quoted literal selection").clone()),
        TextSelection::Input(index) => {
            let text = &input.data[index];
            if text.has_retained_claim() {
                Ok(text.clone())
            } else {
                receipt
                    .with_output_construction(runtime, |permit| {
                        text.try_copy_exact_with(permit, |_| receipt.checkpoint(runtime))
                    })?
                    .map_err(initialization_error)
            }
        }
    }
}

fn ordinary(
    input: StringArray,
    literal: TextLiteral<'_>,
    select: impl Fn(&StringArray, usize) -> TextSelection,
    invalid: impl Fn(String) -> RuntimeError,
) -> BuiltinResult<StringArray> {
    let literal = match literal {
        TextLiteral::Utf8(text) => text.to_owned(),
        TextLiteral::Chars(chars) => chars.iter().collect(),
    };
    let data = (0..input.data.len())
        .map(|index| match select(&input, index) {
            TextSelection::Input(source) => input.data[source].clone(),
            TextSelection::Literal => Utf8Text::from(literal.clone()),
        })
        .collect();
    StringArray::from_texts_unbounded(data, input.shape.as_slice().to_vec()).map_err(invalid)
}
