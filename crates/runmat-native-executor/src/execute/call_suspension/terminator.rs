use runmat_native_codegen::NativeBlockId;
use runmat_runtime::execution::RootedValue;
use runmat_runtime::native::{
    NativeSiteOutcome, NativeSitePhase, NativeSiteRequest, NativeValueRef,
};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{ActiveForLoop, HostState};
use super::payload::{PayloadBox, PayloadReservation};
use super::{CallCompletion, CompletedPayload, PendingCall};

pub(in crate::execute) struct PendingTerminator {
    stage: TerminatorCompletion,
    // Inline metadata; the first physical block's exact layout includes it.
    // The receipt remains active across first-block retirement and Ready allocation.
    reservation: PayloadReservation,
}

impl PendingTerminator {
    pub(super) fn is_for_next(&self) -> bool {
        self.stage.is_for_next()
    }

    pub(super) fn restore_for_next(
        self,
        state: &mut HostState,
    ) -> NativeExecutorResult<(
        Result<Option<RootedValue>, runmat_runtime::RuntimeError>,
        PayloadReservation,
    )> {
        let next = self.stage.restore_for_next(state)?;
        Ok((next, self.reservation))
    }

    pub(super) fn finish(
        self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> NativeExecutorResult<PayloadBox<TerminatorReady>> {
        self.reservation.allocate(runtime, self.stage.finish()?)
    }
}

pub(in crate::execute) fn prepare_payload(
    runtime: &runmat_runtime::context::RuntimeContext,
) -> NativeExecutorResult<PayloadReservation> {
    PayloadReservation::prepare_sequence::<PendingTerminator, TerminatorReady>(runtime)
}

pub(in crate::execute) fn completion(
    runtime: &runmat_runtime::context::RuntimeContext,
    reservation: PayloadReservation,
    stage: TerminatorCompletion,
) -> NativeExecutorResult<CallCompletion> {
    let allocator = reservation.allocator(runtime)?;
    let result = allocator
        .allocate(PendingTerminator { stage, reservation })
        .map_err(|cause| super::payload::error(runtime, cause))?;
    result.reservation.check(runtime)?;
    Ok(CallCompletion::Terminator(result))
}

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
    ForCapture(runmat_runtime::iteration::ForColumnIterator),
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
    ForCapture(runmat_runtime::iteration::ForColumnIterator),
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

    pub(super) fn arena_roots(&self) -> &[NativeValueRef] {
        match &self.stage {
            TerminatorReadyStage::ForNext(next) => next.as_slice(),
            _ => &[],
        }
    }
}

impl TerminatorCompletion {
    pub(super) fn is_for_next(&self) -> bool {
        matches!(self, Self::ForNext { .. })
    }

    pub(super) fn restore_for_next(
        self,
        state: &mut HostState,
    ) -> NativeExecutorResult<Result<Option<RootedValue>, runmat_runtime::RuntimeError>> {
        let Self::ForNext {
            header,
            active,
            next,
        } = self
        else {
            return Err(NativeExecutorError::Host(
                "native terminator completion is not a for iteration".into(),
            ));
        };
        state.restore_for_loop(header, active)?;
        Ok(next)
    }

    pub(super) fn finish(self) -> NativeExecutorResult<TerminatorReady> {
        Ok(match self {
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
    Ok(Some(stage.into_inner()))
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

fn install<F>(
    state: &mut HostState,
    identity: (NativeSiteRequest, u64, u64),
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<NativeSiteOutcome>
where
    F: std::future::Future<Output = NativeExecutorResult<CallCompletion>> + 'static,
{
    let future = super::admit_completion_with(state, make_future)?;
    install_admitted(state, identity, future)
}

fn install_admitted(
    state: &mut HostState,
    identity: (NativeSiteRequest, u64, u64),
    future: super::PendingCompletionFuture,
) -> NativeExecutorResult<NativeSiteOutcome> {
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
    let payload = prepare_payload(&state.runtime)?;
    let condition = RootedValue::new(condition)
        .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    let runtime = state.runtime.clone();
    install(state, identity, move || async move {
        let truth = runtime
            .scope(runmat_runtime::condition::control_flow_truth_from_value(
                &runtime,
                condition.value(),
                "branch condition",
            ))
            .await?;
        completion(&runtime, payload, TerminatorCompletion::BranchTruth(truth))
    })
}

pub(in crate::execute) fn begin_switch_equality(
    state: &mut HostState,
    index: usize,
    discriminant: RootedValue,
    case: Value,
) -> NativeExecutorResult<NativeSiteOutcome> {
    let identity = admission(state)?;
    let payload = prepare_payload(&state.runtime)?;
    let case = RootedValue::new(case)
        .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    let runtime = state.runtime.clone();
    install(state, identity, move || async move {
        let equality = runtime
            .scope(runmat_runtime::call_builtin_async_in_context(
                &runtime,
                "eq",
                &[discriminant.value().clone(), case.value().clone()],
            ))
            .await?;
        completion(
            &runtime,
            payload,
            TerminatorCompletion::SwitchEquality {
                index,
                discriminant,
                equality,
            },
        )
    })
}

pub(in crate::execute) fn begin_switch_truth(
    state: &mut HostState,
    index: usize,
    discriminant: RootedValue,
    equality: RootedValue,
) -> NativeExecutorResult<NativeSiteOutcome> {
    let identity = admission(state)?;
    let payload = prepare_payload(&state.runtime)?;
    let runtime = state.runtime.clone();
    install(state, identity, move || async move {
        let truth = runtime
            .scope(runmat_runtime::condition::logical_truth_from_value(
                equality.value(),
                "switch case comparison",
            ))
            .await?;
        completion(
            &runtime,
            payload,
            TerminatorCompletion::SwitchTruth {
                index,
                discriminant,
                truth,
            },
        )
    })
}

pub(in crate::execute) fn begin_for_capture(
    state: &mut HostState,
    source: Value,
) -> NativeExecutorResult<NativeSiteOutcome> {
    let identity = admission(state)?;
    let payload = prepare_payload(&state.runtime)?;
    let source = RootedValue::new(source)
        .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    let runtime = state.runtime.clone();
    install(state, identity, move || async move {
        let iterator = runtime
            .scope(runmat_runtime::iteration::ForColumnIterator::new(
                source.value().clone(),
            ))
            .await?;
        completion(
            &runtime,
            payload,
            TerminatorCompletion::ForCapture(iterator),
        )
    })
}

pub(in crate::execute) fn begin_for_next(
    state: &mut HostState,
    header: NativeBlockId,
) -> NativeExecutorResult<NativeSiteOutcome> {
    // Validate the handoff before moving the only loop record out of HostState.
    let identity = admission(state)?;
    if !state.has_for_loop(header) {
        return Err(NativeExecutorError::Host(
            "native for-loop state is unavailable".into(),
        ));
    }
    let payload = prepare_payload(&state.runtime)?;
    let runtime = state.runtime.clone();
    let catch = state.active_catch_exception();
    let future = super::admit_completion_in_context_with(&runtime.clone(), catch, || {
        // The exact allocation is admitted before this factory removes the
        // move-only iterator. Refusal therefore leaves the loop in HostState.
        let active = state.take_for_loop(header);
        async move {
            let mut active = active?;
            let next = runtime.scope(active.iterator.next()).await;
            completion(
                &runtime,
                payload,
                TerminatorCompletion::ForNext {
                    header,
                    active,
                    next,
                },
            )
        }
    })?;
    install_admitted(state, identity, future)
}
