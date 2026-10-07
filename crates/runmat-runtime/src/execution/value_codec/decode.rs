use runmat_execution::value::{
    ValueLimits, ValuePayload, ValuePayloadValidationCursor, ValueValidationProgress,
};
use runmat_value::{
    CellArray, CharArray, Closure, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
    LogicalArray, MException, ObjectInstance, SparseTensor, StringArray, StructValue, Tensor,
    Value,
};
use std::num::NonZeroUsize;
use std::rc::Rc;

use crate::execution::{
    CpuCooperationBudget, ExecutionServiceError, RootedValues, RootedValuesBuilder,
};

use super::ValueCodecError;

#[path = "decode/machine.rs"]
mod machine;
use machine::{DecodeMachine, DecodeProgress};

const DECODE_WORK_QUANTUM: NonZeroUsize = NonZeroUsize::new(8 * 1024).unwrap();

/// Scheduling/invocation failure from the cooperative decoder. Portable value
/// errors remain typed and are never flattened into execution diagnostics.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CooperativeValueCodecError {
    #[error(transparent)]
    Value(#[from] ValueCodecError),
    #[error(transparent)]
    Execution(#[from] ExecutionServiceError),
}

pub fn decode_inline_value(payload: &ValuePayload) -> Result<Value, ValueCodecError> {
    payload
        .validate(ValueLimits::default())
        .map_err(|error| ValueCodecError::invalid("$", error.to_string()))?;
    drain_machine(payload)
}

pub(super) fn decode_durable_inline_value(
    payload: &ValuePayload,
) -> Result<Value, ValueCodecError> {
    payload
        .validate_for_durable_storage(ValueLimits::default())
        .map_err(|error| ValueCodecError::invalid("$", error.to_string()))?;
    drain_machine(payload)
}

/// Validate and decode one aggregate under a shared limit and the selected
/// invocation's cooperation budget. The synchronous codec drains this same
/// canonical validation cursor without installing a scheduler policy.
pub async fn decode_inline_values_cooperatively(
    payloads: &[ValuePayload],
    limits: ValueLimits,
    source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
    mut cooperation: Option<&mut CpuCooperationBudget>,
) -> Result<RootedValues, CooperativeValueCodecError> {
    validate_values_cooperatively(
        ValuePayloadValidationCursor::new(payloads, limits),
        &mut cooperation,
    )
    .await?;
    decode_validated_values(payloads, source_resolver, &mut cooperation).await
}

/// Decode a complete durable batch under one shared admission budget and the
/// selected invocation's cooperation policy.
pub async fn decode_durable_values_cooperatively(
    payloads: &[ValuePayload],
    limits: ValueLimits,
    mut cooperation: Option<&mut CpuCooperationBudget>,
) -> Result<RootedValues, CooperativeValueCodecError> {
    for payload in payloads {
        super::durable::require_durable(payload, "$")?;
    }
    validate_values_cooperatively(
        ValuePayloadValidationCursor::for_durable_storage(payloads, limits),
        &mut cooperation,
    )
    .await?;
    decode_validated_values(payloads, None, &mut cooperation).await
}

async fn validate_values_cooperatively(
    mut cursor: ValuePayloadValidationCursor<'_>,
    cooperation: &mut Option<&mut CpuCooperationBudget>,
) -> Result<(), CooperativeValueCodecError> {
    loop {
        let progress = cursor
            .advance(DECODE_WORK_QUANTUM)
            .map_err(|error| ValueCodecError::invalid("$", error.to_string()))?;
        let (completed_work, complete) = match progress {
            ValueValidationProgress::Pending { completed_work } => (completed_work, false),
            ValueValidationProgress::Complete { completed_work } => (completed_work, true),
        };
        if let Some(budget) = cooperation.as_deref_mut() {
            budget.checkpoint(completed_work as u64).await?;
        }
        if complete {
            return Ok(());
        }
    }
}

async fn decode_validated_values(
    payloads: &[ValuePayload],
    source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
    cooperation: &mut Option<&mut CpuCooperationBudget>,
) -> Result<RootedValues, CooperativeValueCodecError> {
    let mut values = RootedValuesBuilder::with_capacity(payloads.len());
    for payload in payloads {
        let mut machine = DecodeMachine::new(payload, "$".to_owned(), source_resolver.clone());
        loop {
            match machine.advance(DECODE_WORK_QUANTUM)? {
                DecodeProgress::Pending { completed_work } => {
                    if let Some(budget) = cooperation.as_deref_mut() {
                        budget.checkpoint(completed_work as u64).await?;
                    }
                }
                DecodeProgress::Complete {
                    value,
                    completed_work,
                } => {
                    if let Some(budget) = cooperation.as_deref_mut() {
                        budget.checkpoint(completed_work as u64).await?;
                    }
                    values.push_rooted(value);
                    break;
                }
            }
        }
    }
    Ok(values.finish()?)
}

fn drain_machine(payload: &ValuePayload) -> Result<Value, ValueCodecError> {
    let mut machine = DecodeMachine::new(payload, "$".to_owned(), None);
    loop {
        match machine.advance(NonZeroUsize::new(usize::MAX).expect("usize::MAX is nonzero"))? {
            DecodeProgress::Pending { .. } => continue,
            DecodeProgress::Complete { value, .. } => return Ok(value.into_unrooted_value()),
        }
    }
}

#[cfg(test)]
#[path = "decode/cooperation_tests.rs"]
mod cooperation_tests;
