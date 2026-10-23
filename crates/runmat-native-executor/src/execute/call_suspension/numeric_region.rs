//! The existing native hostcall slot owns one selected numeric region future.
use super::super::state::HostState;
use super::{CallCompletion, CompletedPayload, PendingCall};
use crate::region::{NumericRegionReferences, OptimizedRegionPlanLease};
use crate::{NativeExecutorError, NativeExecutorResult};
use runmat_runtime::native::NativeSiteRequest;
use runmat_runtime::numeric_region::service::{
    NativeNumericRegionOutcome, NativeNumericRegionRequest,
};

pub(in crate::execute) struct NumericRegionCompletion {
    pub plan: OptimizedRegionPlanLease,
    pub inputs: NumericRegionReferences,
    pub outcome: NativeNumericRegionOutcome,
}

impl NumericRegionCompletion {
    fn into_payload(self: Box<Self>) -> Self {
        *self
    }
}

/// Retire the actual completion Box before its admitting receipt, including
/// abandonment while it is still a tagged pending/completed call payload.
pub(in crate::execute) struct NumericRegionCompletionOwner {
    completion: Box<NumericRegionCompletion>,
    metadata: Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
}

impl NumericRegionCompletionOwner {
    pub(super) fn completion(&self) -> &NumericRegionCompletion {
        &self.completion
    }

    fn into_completion(self) -> NumericRegionCompletion {
        let Self {
            completion,
            metadata,
        } = self;
        // The consuming Box receiver returns only after its backing has been
        // deallocated. The external receipt remains live through that call.
        let completion = completion.into_payload();
        drop(metadata);
        completion
    }
}

pub(in crate::execute) fn take(
    state: &mut HostState,
    request: NativeSiteRequest,
) -> NativeExecutorResult<Option<NumericRegionCompletion>> {
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    if !matches!(&completed.payload, CompletedPayload::NumericRegion(_)) {
        state.completed_call = Some(completed);
        return Ok(None);
    }
    if completed.request != request || completed.embedded.is_some() {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "numeric region resumed at a different native site".into(),
        ));
    }
    let CompletedPayload::NumericRegion(completed) = completed.payload else {
        unreachable!()
    };
    Ok(Some(completed.into_completion()))
}

pub(in crate::execute) fn begin(
    state: &mut HostState,
    plan: OptimizedRegionPlanLease,
    inputs: NumericRegionReferences,
    request: NativeNumericRegionRequest,
) -> NativeExecutorResult<()> {
    if state.pending_call.is_some() || state.completed_call.is_some() {
        return Err(NativeExecutorError::Host(
            "numeric region cannot replace an existing hostcall".into(),
        ));
    }
    let site = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("numeric region has no active native site".into())
    })?;
    let runtime = state.runtime.clone();
    let service = runtime
        .service_ports()
        .acceleration()
        .cloned()
        .ok_or_else(|| {
            NativeExecutorError::Host("numeric region acceleration service disappeared".into())
        })?;
    use runmat_runtime::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
    // Inline reference slots are part of this exact boxed completion layout;
    // no prior Vec capacity or deep plan clone needs retroactive accounting.
    let bytes = u64::try_from(std::mem::size_of::<NumericRegionCompletion>())
        .ok()
        .ok_or_else(|| {
            NativeExecutorError::Host("native region continuation quote overflow".into())
        })?;
    let allocation_error = |source| {
        NativeExecutorError::from(
            runmat_runtime::context::host_allocation::host_allocation_runtime_error(
                Some(&runtime),
                "RunMat:Native:NumericRegion",
                "native region continuation admission failed",
                source,
            ),
        )
    };
    let metadata = HostAllocationReceipt::admit(
        Some(&runtime),
        HostAllocationPlan::new(bytes, 0).map_err(allocation_error)?,
    )
    .map_err(allocation_error)?;
    let admission = runtime.clone();
    let catch = state.active_catch_exception();
    let future = runmat_runtime::execution::admit_native_hostcall_with(
        &admission,
        allocation_cause,
        || {
            runtime
                .clone()
                .scope_active_catch_unboxed_shared(catch, async move {
                    let outcome = service.execute_numeric_region(runtime, request)?.await?;
                    Ok(CallCompletion::NumericRegion(
                        NumericRegionCompletionOwner {
                            completion: Box::new(NumericRegionCompletion {
                                plan,
                                inputs,
                                outcome,
                            }),
                            metadata,
                        },
                    ))
                })
        },
    )?;
    let (continuation, generation) = state.next_suspension_identity()?;
    state.pending_call = Some(PendingCall {
        continuation,
        generation,
        request: site,
        embedded: None,
        // This capability accepts only closed real-floating owners; their
        // backing is in request, with no GC handles. Original generation references
        // remain in the frame and inline completion, not a duplicate root Vec.
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
        "RunMat:Native:NumericRegion",
        "native region continuation admission failed",
        source,
    )
    .into()
}

pub(in crate::execute) fn begin_original_completion(
    state: &mut HostState,
    request: NativeSiteRequest,
    completion: runmat_runtime::execution::RuntimeContinuationFuture<'static, ()>,
) -> NativeExecutorResult<()> {
    if state.pending_call.is_some() || state.completed_call.is_some() {
        return Err(NativeExecutorError::Host(
            "original region completion cannot replace a hostcall".into(),
        ));
    }
    let Ok(future) = runmat_runtime::execution::admit_native_hostcall_with(
        &state.runtime,
        allocation_cause,
        || async move {
            // The domain observation owner suppresses optional tail/sample errors;
            // source stores were already published and must never be replayed.
            let _ = completion.await;
            Ok(CallCompletion::OriginalRegionObserved)
        },
    ) else {
        return Ok(());
    };
    let (continuation, generation) = state.next_suspension_identity()?;
    state.pending_call = Some(PendingCall {
        continuation,
        generation,
        request,
        embedded: None,
        retained_roots: Vec::new(),
        future,
    });
    Err(NativeExecutorError::CallSuspended)
}

pub(in crate::execute) fn take_original_completion(
    state: &mut HostState,
    request: NativeSiteRequest,
) -> NativeExecutorResult<()> {
    let Some(completed) = state.completed_call.as_ref() else {
        return Ok(());
    };
    if !matches!(completed.payload, CompletedPayload::OriginalRegionObserved) {
        return Ok(());
    }
    if completed.request != request || completed.embedded.is_some() {
        return Err(NativeExecutorError::Host(
            "original native region observation resumed at a different site".into(),
        ));
    }
    state.completed_call = None;
    Ok(())
}
