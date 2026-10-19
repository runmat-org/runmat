use super::{ValueLimits, ValuePayload, ValueRef, ValueTransportContext};
use crate::ContractError;
use std::num::NonZeroUsize;

mod cursor;
pub use cursor::{ValuePayloadValidationCursor, ValueValidationProgress};

pub(super) fn validate(value: &ValuePayload, limits: ValueLimits) -> Result<(), ContractError> {
    drain(ValuePayloadValidationCursor::new(
        std::slice::from_ref(value),
        limits,
    ))
}
pub(super) fn validate_reference(
    reference: &ValueRef,
    limits: ValueLimits,
) -> Result<(), ContractError> {
    validate(&ValuePayload::Object(Box::new(reference.clone())), limits)
}
pub(super) fn validate_all(
    values: &[ValuePayload],
    limits: ValueLimits,
) -> Result<(), ContractError> {
    drain(ValuePayloadValidationCursor::new(values, limits))
}
pub(super) fn validate_graph_leaves<'a>(
    leaves: &[(&'a ValuePayload, u16)],
    limits: ValueLimits,
    sources: Option<&'a crate::ExecutableSourceMapV2>,
) -> Result<(), ContractError> {
    drain(ValuePayloadValidationCursor::for_graph_leaves(
        leaves, limits, sources,
    ))
}
pub(super) fn validate_for_program_sources<'a>(
    value: &'a ValuePayload,
    limits: ValueLimits,
    sources: Option<&'a crate::ExecutableSourceMapV2>,
) -> Result<(), ContractError> {
    drain(ValuePayloadValidationCursor::for_program_sources(
        std::slice::from_ref(value),
        limits,
        sources,
    ))
}
pub(super) fn validate_for_graph_sources<'a>(
    value: &'a ValuePayload,
    limits: ValueLimits,
    sources: Option<&'a crate::ExecutableSourceMapV2>,
) -> Result<(), ContractError> {
    drain(ValuePayloadValidationCursor::for_graph_sources(
        std::slice::from_ref(value),
        limits,
        sources,
    ))
}
pub(super) fn validate_for_durable_storage(
    value: &ValuePayload,
    limits: ValueLimits,
) -> Result<(), ContractError> {
    drain(ValuePayloadValidationCursor::for_durable_storage(
        std::slice::from_ref(value),
        limits,
    ))
}
pub(super) fn validate_all_for_durable_storage(
    values: &[ValuePayload],
    limits: ValueLimits,
) -> Result<(), ContractError> {
    drain(ValuePayloadValidationCursor::for_durable_storage(
        values, limits,
    ))
}
pub(super) fn validate_for_transport<'a>(
    value: &'a ValuePayload,
    limits: ValueLimits,
    context: ValueTransportContext<'a>,
) -> Result<(), ContractError> {
    drain(ValuePayloadValidationCursor::for_transport(
        std::slice::from_ref(value),
        limits,
        context,
    ))
}
fn drain(mut cursor: ValuePayloadValidationCursor<'_>) -> Result<(), ContractError> {
    let maximum = NonZeroUsize::new(usize::MAX).expect("usize::MAX is nonzero");
    loop {
        match cursor.advance(maximum)? {
            ValueValidationProgress::Pending { .. } => {}
            ValueValidationProgress::Complete { .. } => return Ok(()),
        }
    }
}
