use runmat_native_codegen::NativeBlockId;
use runmat_runtime::execution::RootedValue;
use runmat_runtime::native::{
    NativeSiteOutcome, NativeSitePhase, NativeSiteRequest, NativeValueRef,
};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{ActiveForLoop, HostState};
use super::{CallCompletion, CompletedPayload, CompletionFuture, PendingCall};

pub(in crate::execute) enum TerminatorCompletion {
    BranchTruth(bool),
    SwitchEquality {
        index: usize,
        discriminant: RootedValue,
        equality: RootedValue,
    },
    SwitchTruth {
        index: usize,
        discriminant: RootedValue,
        truth: bool,
    },
    ForCapture(Box<runmat_runtime::iteration::ForColumnIterator>),
    ForNext {
        header: NativeBlockId,
        active: ActiveForLoop,
        next: Result<Option<RootedValue>, runmat_runtime::RuntimeError>,
    },
}

/// The existing boxed completion owns this record; separating the second
/// rooted switch operand keeps the stage tag compact without allocating or
/// releasing either value before the resumed site consumes it.
pub(in crate::execute) struct TerminatorReady {
    stage: TerminatorReadyStage,
    equality: Option<RootedValue>,
}

pub(in crate::execute) enum TerminatorReadyStage {
    BranchTruth(bool),
    SwitchEquality {
        index: usize,
        discriminant: RootedValue,
    },
    SwitchTruth {
        index: usize,
        discriminant: RootedValue,
        truth: bool,
    },
    ForCapture(Box<runmat_runtime::iteration::ForColumnIterator>),
    ForNext(Option<NativeValueRef>),
}

impl TerminatorReady {
    pub(super) fn for_next(next: Option<NativeValueRef>) -> Self {
        Self {
            stage: TerminatorReadyStage::ForNext(next),
            equality: None,
        }
    }

    pub(in crate::execute) fn into_parts(self) -> (TerminatorReadyStage, Option<RootedValue>) {
        (self.stage, self.equality)
    }

    pub(super) fn arena_roots(&self) -> Vec<NativeValueRef> {
        match &self.stage {
            TerminatorReadyStage::ForNext(Some(reference)) => vec![*reference],
            _ => Vec::new(),
        }
    }
}

impl TerminatorCompletion {
    pub(super) fn is_for_next(&self) -> bool {
        matches!(self, Self::ForNext { .. })
    }

    pub(super) fn restore_for_next(
        self: Box<Self>,
        state: &mut HostState,
    ) -> NativeExecutorResult<Result<Option<RootedValue>, runmat_runtime::RuntimeError>> {
        let Self::ForNext {
            header,
            active,
            next,
        } = *self
        else {
            return Err(NativeExecutorError::Host(
                "native terminator completion is not a for iteration".into(),
            ));
        };
        state.restore_for_loop(header, active)?;
        Ok(next)
    }

    pub(super) fn finish(self: Box<Self>) -> NativeExecutorResult<TerminatorReady> {
        Ok(match *self {
            Self::BranchTruth(truth) => TerminatorReady {
                stage: TerminatorReadyStage::BranchTruth(truth),
                equality: None,
            },
            Self::SwitchEquality {
                index,
                discriminant,
                equality,
            } => TerminatorReady {
                stage: TerminatorReadyStage::SwitchEquality {
                    index,
                    discriminant,
                },
                equality: Some(equality),
            },
            Self::SwitchTruth {
                index,
                discriminant,
                truth,
            } => TerminatorReady {
                stage: TerminatorReadyStage::SwitchTruth {
                    index,
                    discriminant,
                    truth,
                },
                equality: None,
            },
            Self::ForCapture(iterator) => TerminatorReady {
                stage: TerminatorReadyStage::ForCapture(iterator),
                equality: None,
            },
            Self::ForNext { .. } => {
                return Err(NativeExecutorError::Host(
                    "for iteration completed without restoring its owner".into(),
                ));
            }
        })
    }
}

pub(in crate::execute) fn take_completed_terminator(
    state: &mut HostState,
) -> NativeExecutorResult<Option<TerminatorReady>> {
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed native terminator has no active site".into())
    })?;
    if completed.request != request || completed.embedded.is_some() {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed native terminator does not match the exact resumed site".into(),
        ));
    }
    if !matches!(&completed.payload, CompletedPayload::Terminator(..)) {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed native terminator has the wrong typed payload".into(),
        ));
    }
    let CompletedPayload::Terminator(stage) = completed.payload else {
        unreachable!("typed terminator payload was checked")
    };
    Ok(Some(*stage))
}

fn admission(state: &mut HostState) -> NativeExecutorResult<(NativeSiteRequest, u64, u64)> {
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending native terminator has no active site".into())
    })?;
    if request.phase != NativeSitePhase::TERMINATOR {
        return Err(NativeExecutorError::Host(
            "native terminator suspension has the wrong site phase".into(),
        ));
    }
    let (continuation, generation) = state.next_suspension_identity()?;
    Ok((request, continuation, generation))
}

fn install(
    state: &mut HostState,
    identity: (NativeSiteRequest, u64, u64),
    future: CompletionFuture,
) -> NativeExecutorResult<NativeSiteOutcome> {
    let future = super::scope_completion(state, future);
    state.pending_call = Some(PendingCall {
        request: identity.0,
        continuation: identity.1,
        generation: identity.2,
        embedded: None,
        retained_roots: Vec::new(),
        future,
    });
    Err(NativeExecutorError::CallSuspended)
}

pub(in crate::execute) fn begin_branch_truth(
    state: &mut HostState,
    condition: Value,
) -> NativeExecutorResult<NativeSiteOutcome> {
    let identity = admission(state)?;
    let condition = RootedValue::new(condition)
        .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    let runtime = state.runtime.clone();
    install(
        state,
        identity,
        Box::pin(async move {
            let truth = runtime
                .scope(runmat_runtime::condition::logical_truth_from_value(
                    condition.value(),
                    "branch condition",
                ))
                .await?;
            Ok(CallCompletion::Terminator(Box::new(
                TerminatorCompletion::BranchTruth(truth),
            )))
        }),
    )
}

pub(in crate::execute) fn begin_switch_equality(
    state: &mut HostState,
    index: usize,
    discriminant: RootedValue,
    case: Value,
) -> NativeExecutorResult<NativeSiteOutcome> {
    let identity = admission(state)?;
    let case = RootedValue::new(case)
        .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    let runtime = state.runtime.clone();
    install(
        state,
        identity,
        Box::pin(async move {
            let equality = runtime
                .scope(runmat_runtime::call_builtin_async_in_context(
                    &runtime,
                    "eq",
                    &[discriminant.value().clone(), case.value().clone()],
                ))
                .await?;
            Ok(CallCompletion::Terminator(Box::new(
                TerminatorCompletion::SwitchEquality {
                    index,
                    discriminant,
                    equality,
                },
            )))
        }),
    )
}

pub(in crate::execute) fn begin_switch_truth(
    state: &mut HostState,
    index: usize,
    discriminant: RootedValue,
    equality: RootedValue,
) -> NativeExecutorResult<NativeSiteOutcome> {
    let identity = admission(state)?;
    let runtime = state.runtime.clone();
    install(
        state,
        identity,
        Box::pin(async move {
            let truth = runtime
                .scope(runmat_runtime::condition::logical_truth_from_value(
                    equality.value(),
                    "switch case comparison",
                ))
                .await?;
            Ok(CallCompletion::Terminator(Box::new(
                TerminatorCompletion::SwitchTruth {
                    index,
                    discriminant,
                    truth,
                },
            )))
        }),
    )
}

pub(in crate::execute) fn begin_for_capture(
    state: &mut HostState,
    source: Value,
) -> NativeExecutorResult<NativeSiteOutcome> {
    let identity = admission(state)?;
    let source = RootedValue::new(source)
        .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    let runtime = state.runtime.clone();
    install(
        state,
        identity,
        Box::pin(async move {
            let iterator = runtime
                .scope(runmat_runtime::iteration::ForColumnIterator::new(
                    source.value().clone(),
                ))
                .await?;
            Ok(CallCompletion::Terminator(Box::new(
                TerminatorCompletion::ForCapture(Box::new(iterator)),
            )))
        }),
    )
}

pub(in crate::execute) fn begin_for_next(
    state: &mut HostState,
    header: NativeBlockId,
) -> NativeExecutorResult<NativeSiteOutcome> {
    // Validate the handoff before moving the only loop record out of HostState.
    let identity = admission(state)?;
    let mut active = state.take_for_loop(header)?;
    let runtime = state.runtime.clone();
    install(
        state,
        identity,
        Box::pin(async move {
            let next = runtime.scope(active.iterator.next()).await;
            Ok(CallCompletion::Terminator(Box::new(
                TerminatorCompletion::ForNext {
                    header,
                    active,
                    next,
                },
            )))
        }),
    )
}
