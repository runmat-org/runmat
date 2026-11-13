//! Allocation authority for operation bookkeeping and callback-held funding.
use super::*;
use runmat_execution::resource::RetainedHostAllocationClaim;

impl ProviderOperationRecorder {
    /// Clone the already-admitted aggregate funding for an outer callback
    /// owner. This is lifetime retention, not authority for new allocation.
    pub fn completion_host_metadata_claim(&self) -> Option<Arc<dyn RetainedHostAllocationClaim>> {
        self.0.with(|state| state.operation_host_metadata.clone())
    }
}

/// Payload, strong/weak counters and alignment padding for an Arc owner.
/// Handles cloned into callbacks share this allocation.
pub(super) fn arc_host_upper_bytes<T>() -> Option<u64> {
    std::mem::size_of::<T>()
        .checked_add(std::mem::size_of::<usize>().checked_mul(2)?)?
        .checked_add(std::mem::align_of::<T>())?
        .try_into()
        .ok()
}

impl ProviderOperationReceipt {
    /// New owners allocated by `with_budget` (also used by WGPU progress).
    /// The caller already owns the shared budget and optional progress port.
    /// This includes inline resource storage, but not additional scopes,
    /// operation-specific strings, tickets or materialized snapshots.
    pub fn with_budget_host_metadata_upper_bytes() -> Option<u64> {
        arc_host_upper_bytes::<Shared>()?.checked_add(arc_host_upper_bytes::<Clock>()?)
    }

    /// Retain pre-admitted operation metadata through exact callback lifetime.
    /// The claim must not own this receipt or any recorder derived from it.
    /// A single operation authority aggregates its funding before effects;
    /// replacing funding after submission would release it prematurely.
    pub fn retain_operation_host_metadata(
        &mut self,
        claim: Arc<dyn RetainedHostAllocationClaim>,
    ) -> Result<(), ProviderObservationFailure> {
        let mut claim = Some(claim);
        self.0.with(|state| {
            if let Some(failure) = state.failure {
                return Err(failure);
            }
            if state.operation_host_metadata.is_some()
                || state.registered != 0
                || state.resources.has_issued_resources()
                || state.publication.is_some()
                || state.taken
            {
                state.fail(ProviderObservationFailure::TrackingOverflow);
                return Err(ProviderObservationFailure::TrackingOverflow);
            }
            state.operation_host_metadata = claim.take();
            Ok(())
        })
    }

    /// Non-consuming barrier for serial resource phases. Unlike a resource
    /// snapshot this does not allocate diagnostic vectors or finish the scope.
    /// Failure is returned only after all issued tickets have retired.
    pub fn poll_resource_quiescence(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), ProviderObservationFailure>> {
        let mut next = Some(cx.waker().clone());
        let (result, old) = self.0.with(|state| {
            if state.resources.live_count() != 0 {
                (
                    Poll::Pending,
                    std::mem::replace(&mut state.resource_waiter, next.take()),
                )
            } else {
                (Poll::Ready(state.failure.map_or(Ok(()), Err)), None)
            }
        });
        drop(old);
        result
    }
}
