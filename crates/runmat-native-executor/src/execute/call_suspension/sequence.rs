use std::future::Future;

use runmat_runtime::sequence::{PreparedSequenceDestination, SequenceDestinationBuilder};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{EmbeddedOperationIdentity, HostState};
use super::{CallCompletion, CompletedPayload, PendingCall};

#[path = "sequence/end.rs"]
mod end;
pub(in crate::execute) use end::{
    begin_sequence_end_stage, SequenceEndAdmission, SequenceEndOwner,
};

pub(in crate::execute) enum SequenceStage {
    Builder(SequenceDestinationBuilder),
    Destination(PreparedSequenceDestination),
    End(SequenceEndOwner),
}

pub(in crate::execute) fn take_completed_sequence_stage(
    state: &mut HostState,
    embedded: Option<&EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Option<SequenceStage>> {
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed sequence preparation has no active site".into())
    })?;
    if completed.request != request || completed.embedded.as_ref() != embedded {
        let descendant = completed.request == request
            && completed
                .embedded
                .as_ref()
                .is_some_and(|identity| identity.is_descendant_of(embedded));
        state.completed_call = Some(completed);
        if descendant {
            return Ok(None);
        }
        return Err(NativeExecutorError::Host(
            "completed sequence preparation does not match its exact resumed operation".into(),
        ));
    }
    if !matches!(&completed.payload, CompletedPayload::SequenceStage(_)) {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed sequence preparation has the wrong typed payload".into(),
        ));
    }
    if let Some(embedded) = embedded {
        state.restore_embedded_call_regions(embedded);
    }
    match completed.payload {
        CompletedPayload::SequenceStage(stage) => Ok(Some(stage)),
        _ => unreachable!("validated owned stage payload"),
    }
}

pub(in crate::execute) fn begin_sequence_stage<F>(
    state: &mut HostState,
    embedded: Option<EmbeddedOperationIdentity>,
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<SequenceStage>
where
    F: Future<Output = NativeExecutorResult<SequenceStage>> + 'static,
{
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending sequence preparation has no active site".into())
    })?;
    let runtime = state.runtime.clone();
    let catch = state.active_catch_exception();
    let future =
        runmat_runtime::execution::admit_native_hostcall_with(&runtime, allocation_cause, || {
            let future = make_future();
            runtime.scope_active_catch_unboxed_shared(catch, async move {
                future.await.map(CallCompletion::SequenceStage)
            })
        })?;
    let (continuation, generation) = state.next_suspension_identity()?;
    state.pending_call = Some(PendingCall {
        continuation,
        generation,
        request,
        embedded,
        retained_roots: Vec::new(),
        future,
    });
    Err(NativeExecutorError::CallSuspended)
}

fn allocation_cause(
    source: runmat_runtime::context::host_allocation::HostAllocationError,
) -> NativeExecutorError {
    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
        None,
        "RunMat:Native:SequencePreparation",
        "sequence preparation continuation admission failed",
        source,
    )
    .into()
}
