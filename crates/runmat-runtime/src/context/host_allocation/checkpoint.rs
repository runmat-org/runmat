//! Read-only identity of an allocation operation's original owner. Capturing
//! this token allocates nothing and grants no bytes, permit, or service.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use runmat_execution::cancellation::CancellationToken;

use super::{HostAllocationContext, HostAllocationError};
use crate::context::{
    host_effect::ManagedHostEffectLedger, RuntimeContext, RuntimeParallelService,
};

#[derive(Clone)]
pub(super) struct AllocationOwnerState {
    pub(super) ledger: ManagedHostEffectLedger,
    pub(super) epoch: u64,
    pub(super) cancellation: CancellationToken,
    pub(super) generation_owner: Arc<AtomicU64>,
    pub(super) generation: u64,
}

impl AllocationOwnerState {
    pub(super) fn check(&self) -> Result<(), HostAllocationError> {
        if self.cancellation.is_cancelled() {
            return Err(HostAllocationError::Cancelled);
        }
        let state = self.ledger.snapshot();
        if self.generation_owner.load(Ordering::Acquire) != self.generation
            || !state.current
            || state.epoch != self.epoch
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(())
    }

    pub(super) fn check_parallel(
        &self,
        parallel: &dyn RuntimeParallelService,
    ) -> Result<(), HostAllocationError> {
        let ledger = parallel
            .managed_host_effect_ledger()
            .ok_or(HostAllocationError::StaleAuthority)?;
        if !ledger.same_owner(&self.ledger) || parallel.placement_resources().epoch != self.epoch {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct HostAllocationCheckpoint {
    pub(super) context: HostAllocationContext,
    pub(super) owner: AllocationOwnerState,
}

/// Read-only original ledger/generation authority for a Send CPU job. This
/// grants no bytes and contains no thread-affine Runtime context or service.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct HostAllocationWorkerCheckpoint(AllocationOwnerState);

#[cfg(not(target_arch = "wasm32"))]
impl HostAllocationWorkerCheckpoint {
    pub(crate) fn from_checkpoint(checkpoint: &HostAllocationCheckpoint) -> Self {
        Self(checkpoint.owner.clone())
    }
    pub(crate) fn check(&self) -> Result<(), HostAllocationError> {
        self.0.check()
    }
}

impl HostAllocationCheckpoint {
    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        self.context.same_owner(&other.context)
            && self.owner.ledger.same_owner(&other.owner.ledger)
            && self.owner.epoch == other.owner.epoch
            && self.owner.generation == other.owner.generation
            && Arc::ptr_eq(&self.owner.generation_owner, &other.owner.generation_owner)
            && self
                .owner
                .cancellation
                .same_generation(&other.owner.cancellation)
    }

    pub(crate) fn check_original(&self) -> Result<(), HostAllocationError> {
        self.check_captured()
    }
    pub(crate) fn check(&self, runtime: &RuntimeContext) -> Result<(), HostAllocationError> {
        self.owner.check()?;
        if !self.context.matches_runtime(runtime)
            || !self.context.matches_cancellation(&self.owner.cancellation)
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        let parallel = runtime
            .service_ports()
            .parallel()
            .ok_or(HostAllocationError::StaleAuthority)?;
        self.owner.check_parallel(parallel.as_ref())?;
        self.owner.check()
    }

    pub(super) fn check_captured(&self) -> Result<(), HostAllocationError> {
        self.owner.check()?;
        if !self.context.matches_cancellation(&self.owner.cancellation) {
            return Err(HostAllocationError::StaleAuthority);
        }
        self.owner.check_parallel(self.context.parallel())?;
        self.owner.check()
    }
}
