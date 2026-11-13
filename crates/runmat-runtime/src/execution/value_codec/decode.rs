use runmat_execution::value::{
    ValueLimits, ValuePayload, ValuePayloadValidationCursor, ValueValidationProgress,
};
use runmat_value::{
    CellArray, CharArray, Closure, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
    LogicalArray, MException, SparseTensor, StringArray, Tensor, Value,
};
use std::num::NonZeroUsize;
use std::rc::Rc;

use crate::execution::{CpuCooperationBudget, ExecutionServiceError, RootedValues};

use super::ValueCodecError;

#[path = "decode/machine.rs"]
mod machine;
use machine::{DecodeMachine, DecodeProgress};
#[path = "decode/completion.rs"]
mod completion;
use completion::DecodedCollection;

const DECODE_WORK_QUANTUM: NonZeroUsize = NonZeroUsize::new(8 * 1024).unwrap();

pub(super) trait InlineFrameSourceAuthority {
    fn frame_source(
        &self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Option<(runmat_types::SourceId, crate::CallFrameSourceOwner)>;
}

impl InlineFrameSourceAuthority for super::graph::GraphSourceIssuer {
    fn frame_source(
        &self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Option<(runmat_types::SourceId, crate::CallFrameSourceOwner)> {
        super::graph::GraphSourceIssuer::frame_source(self, ordinal)
    }
}

/// Scheduling/invocation failure from the cooperative decoder. Portable value
/// errors remain typed and are never flattened into execution diagnostics.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CooperativeValueCodecError {
    #[error("{0}")]
    Value(
        #[from]
        #[source]
        ValueCodecError,
    ),
    #[error("{0}")]
    Execution(
        #[from]
        #[source]
        ExecutionServiceError,
    ),
}

pub fn decode_inline_value(payload: &ValuePayload) -> Result<Value, ValueCodecError> {
    Ok(decode_inline_value_rooted(payload)?.into_unrooted_value())
}

/// Decode a value while preserving construction roots through publication to
/// its completed-result owner. Use this at asynchronous host service boundaries.
pub fn decode_inline_value_owned(
    payload: &ValuePayload,
) -> Result<crate::execution::RootedValue, CooperativeValueCodecError> {
    decode_inline_value_owned_with_limits(payload, ValueLimits::default())
}

pub fn decode_inline_value_owned_with_limits(
    payload: &ValuePayload,
    limits: ValueLimits,
) -> Result<crate::execution::RootedValue, CooperativeValueCodecError> {
    payload
        .validate(limits)
        .map_err(|error| ValueCodecError::invalid("$", error.to_string()))?;
    Ok(drain_machine_rooted(payload)?.into_rooted_value()?)
}

pub(super) fn decode_inline_value_owned_with_frame_sources(
    payload: &ValuePayload,
    limits: ValueLimits,
    sources: &dyn InlineFrameSourceAuthority,
) -> Result<crate::execution::RootedValue, CooperativeValueCodecError> {
    payload
        .validate(limits)
        .map_err(|error| ValueCodecError::invalid("$", error.to_string()))?;
    Ok(drain_machine_rooted_with_frames(payload, Some(sources))?.into_rooted_value()?)
}

pub(crate) fn decode_inline_value_rooted(
    payload: &ValuePayload,
) -> Result<crate::execution::RootedValueMember, ValueCodecError> {
    payload
        .validate(ValueLimits::default())
        .map_err(|error| ValueCodecError::invalid("$", error.to_string()))?;
    drain_machine_rooted(payload)
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
    decode_validated_values(payloads, source_resolver, &mut cooperation, None).await
}

/// Decode into one admitted final argument backing under a selected Runtime.
/// The outer collection and private decoder control blocks have independent
/// admitted owners. Child payloads and construction roots retain their own
/// representation-specific allocation/publication contracts.
pub async fn decode_inline_values_cooperatively_in_context(
    payloads: &[ValuePayload],
    limits: ValueLimits,
    source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
    mut cooperation: Option<&mut CpuCooperationBudget>,
    runtime: &crate::context::RuntimeContext,
) -> Result<RootedValues, CooperativeValueCodecError> {
    validate_values_cooperatively(
        ValuePayloadValidationCursor::new(payloads, limits),
        &mut cooperation,
    )
    .await?;
    decode_validated_values(payloads, source_resolver, &mut cooperation, Some(runtime)).await
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
    decode_validated_values(payloads, None, &mut cooperation, None).await
}

/// Decode one already-stored preference under the selected invocation. This
/// uses the same durable validation and decoder as batch import, but avoids
/// constructing every document value during document admission.
pub(crate) async fn decode_durable_value_cooperatively_in_context(
    payload: &ValuePayload,
    cooperation: &mut CpuCooperationBudget,
    runtime: &crate::context::RuntimeContext,
) -> Result<crate::execution::RootedValueMember, CooperativeValueCodecError> {
    super::durable::require_durable(payload, "$")?;
    let mut budget = Some(cooperation);
    validate_values_cooperatively(
        ValuePayloadValidationCursor::for_durable_storage(
            std::slice::from_ref(payload),
            ValueLimits::default(),
        ),
        &mut budget,
    )
    .await?;
    let mut machine = DecodeMachine::new_with_runtime(payload, "$", None, Some(runtime))?;
    loop {
        let progress = machine.advance(DECODE_WORK_QUANTUM)?;
        let (completed_work, complete) = match progress {
            DecodeProgress::Pending { completed_work } => (completed_work, false),
            DecodeProgress::Complete { completed_work } => (completed_work, true),
        };
        if let Some(budget) = budget.as_deref_mut() {
            budget.checkpoint(completed_work as u64).await?;
        }
        if complete {
            return Ok(machine.take_completed());
        }
    }
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
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<RootedValues, CooperativeValueCodecError> {
    let mut values = DecodedCollection::new(payloads.len(), runtime)?;
    for payload in payloads {
        let mut machine =
            DecodeMachine::new_with_runtime(payload, "$", source_resolver.clone(), runtime)?;
        loop {
            match machine.advance(DECODE_WORK_QUANTUM)? {
                DecodeProgress::Pending { completed_work } => {
                    if let Some(budget) = cooperation.as_deref_mut() {
                        budget.checkpoint(completed_work as u64).await?;
                    }
                    values.checkpoint()?;
                }
                DecodeProgress::Complete { completed_work } => {
                    if let Some(budget) = cooperation.as_deref_mut() {
                        budget.checkpoint(completed_work as u64).await?;
                    }
                    values.checkpoint()?;
                    values.push(machine.take_completed())?;
                    break;
                }
            }
        }
    }
    values.finish()
}

fn drain_machine(payload: &ValuePayload) -> Result<Value, ValueCodecError> {
    Ok(drain_machine_rooted(payload)?.into_unrooted_value())
}

fn drain_machine_rooted(
    payload: &ValuePayload,
) -> Result<crate::execution::RootedValueMember, ValueCodecError> {
    drain_machine_rooted_with_frames(payload, None)
}

fn drain_machine_rooted_with_frames<'a>(
    payload: &'a ValuePayload,
    frame_sources: Option<&'a dyn InlineFrameSourceAuthority>,
) -> Result<crate::execution::RootedValueMember, ValueCodecError> {
    crate::builtins::ensure_builtin_value_classes_registered();
    let mut machine = DecodeMachine::new_with_frame_sources(payload, "$", None, frame_sources)?;
    loop {
        match machine.advance(NonZeroUsize::new(usize::MAX).expect("usize::MAX is nonzero"))? {
            DecodeProgress::Pending { .. } => continue,
            DecodeProgress::Complete { .. } => return Ok(machine.take_completed()),
        }
    }
}

#[cfg(test)]
#[path = "decode/cooperation_tests.rs"]
mod cooperation_tests;
