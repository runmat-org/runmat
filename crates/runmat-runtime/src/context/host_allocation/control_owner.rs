//! Original managed authority for RunMat-owned control allocations.
//! This token contains no RuntimeContext, invocation frame or service port.
use runmat_value::ExactHostAllocationClaim;

use super::{checkpoint::AllocationOwnerState, HostAllocationError, HostAllocationReceipt};
use crate::context::{RuntimeContext, RuntimeContextLocalId};

#[derive(Clone)]
pub struct HostControlAllocationOwner {
    owner: AllocationOwnerState,
    local_identity: RuntimeContextLocalId,
}

impl std::fmt::Debug for HostControlAllocationOwner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostControlAllocationOwner")
            .field("epoch", &self.owner.epoch)
            .finish_non_exhaustive()
    }
}

impl HostControlAllocationOwner {
    /// Recheck the original grant before new background allocation/publication.
    /// Reading already-funded immutable storage does not require this check.
    pub fn check_current(&self) -> Result<(), HostAllocationError> {
        self.owner.check()
    }

    /// An absent owner is the explicit unbounded policy. A finite context
    /// without its managed ledger is an error, not an unbounded substitute.
    pub fn capture(runtime: &RuntimeContext) -> Result<Option<Self>, HostAllocationError> {
        let Some(checkpoint) = HostAllocationReceipt::capture_owner(Some(runtime))? else {
            return Ok(None);
        };
        Ok(Some(Self {
            owner: checkpoint.owner,
            local_identity: runtime.local_identity(),
        }))
    }

    pub fn checkpoint(&self, runtime: &RuntimeContext) -> Result<(), HostAllocationError> {
        self.owner.check()?;
        if runtime.local_identity() != self.local_identity
            || !self
                .owner
                .cancellation
                .same_generation(&runtime.cancellation())
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        let parallel = runtime
            .service_ports()
            .parallel()
            .ok_or(HostAllocationError::StaleAuthority)?;
        if parallel
            .placement_resources()
            .memory_available_bytes
            .is_none()
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        self.owner.check_parallel(parallel.as_ref())?;
        self.owner.check()
    }

    /// Reserve the actual requested block and its canonical claim header.
    /// The allocation owner must retain this claim until physical deallocation.
    pub fn claim(&self, bytes: usize) -> Result<ExactHostAllocationClaim, HostAllocationError> {
        self.owner.check()?;
        let claim = self
            .owner
            .ledger
            .reserve_dynamic_scratch_claim(bytes, self.owner.epoch)?;
        self.owner.check()?;
        Ok(claim)
    }
}
