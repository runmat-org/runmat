//! Dynamically admitted, exact physical traversal scratch for Value tracing.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::cancellation::CancellationToken;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostAllocationReservation,
    ExactHostNumericAllocationError,
};

use super::HostAllocationError;
use crate::context::host_effect::{HostEffectAllocationError, ManagedHostEffectLedger};
use crate::context::{RuntimeContext, RuntimeContextLocalId};

/// One operation's issuer, not a reservation. Each growth reserves the exact
/// frame backing plus the canonical owned-claim header before allocating it.
pub(crate) struct HostTraceScratchPermit {
    ledger: ManagedHostEffectLedger,
    epoch: u64,
    local_identity: RuntimeContextLocalId,
    cancellation: CancellationToken,
    generation_owner: Arc<AtomicU64>,
    generation: u64,
    last_failure: Mutex<Option<HostAllocationError>>,
}

impl fmt::Debug for HostTraceScratchPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HostTraceScratchPermit")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl HostTraceScratchPermit {
    pub(super) fn new(
        runtime: &RuntimeContext,
        ledger: ManagedHostEffectLedger,
        epoch: u64,
    ) -> Result<Self, HostAllocationError> {
        let (generation_owner, generation) = runtime.host_effect_session_generation();
        let permit = Self {
            ledger,
            epoch,
            local_identity: runtime.local_identity(),
            cancellation: runtime.cancellation(),
            generation_owner,
            generation,
            last_failure: Mutex::new(None),
        };
        permit.checkpoint(runtime)?;
        Ok(permit)
    }

    fn check_local(&self) -> Result<(), HostAllocationError> {
        if self.cancellation.is_cancelled() {
            return Err(HostAllocationError::Cancelled);
        }
        let snapshot = self.ledger.snapshot();
        if !snapshot.current
            || snapshot.epoch != self.epoch
            || self.generation_owner.load(Ordering::Acquire) != self.generation
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(())
    }

    pub(crate) fn checkpoint(&self, runtime: &RuntimeContext) -> Result<(), HostAllocationError> {
        self.check_local()?;
        if runtime.local_identity() != self.local_identity
            || !self.cancellation.same_generation(&runtime.cancellation())
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        let parallel = runtime
            .service_ports()
            .parallel()
            .ok_or(HostAllocationError::StaleAuthority)?;
        let resources = parallel.placement_resources();
        let ledger = parallel
            .managed_host_effect_ledger()
            .ok_or(HostAllocationError::StaleAuthority)?;
        if resources.memory_available_bytes.is_none()
            || resources.epoch != self.epoch
            || !ledger.same_owner(&self.ledger)
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        self.check_local()
    }

    /// Recover the typed current-operation cause hidden by the narrow Value
    /// allocator trait's physical-error return type.
    pub(crate) fn source_failure(
        &self,
        error: ExactHostNumericAllocationError,
    ) -> HostAllocationError {
        self.last_failure
            .lock()
            .expect("trace scratch failure lock")
            .take()
            .unwrap_or(HostAllocationError::Numeric(error))
    }

    fn save_failure(&self, error: HostAllocationError) -> ExactHostNumericAllocationError {
        *self
            .last_failure
            .lock()
            .expect("trace scratch failure lock") = Some(error);
        match error {
            HostAllocationError::Overflow
            | HostAllocationError::Resource(HostEffectAllocationError::Overflow) => {
                ExactHostNumericAllocationError::LayoutOverflow
            }
            HostAllocationError::Resource(HostEffectAllocationError::AllocationFailed) => {
                ExactHostNumericAllocationError::AllocationFailed
            }
            _ => ExactHostNumericAllocationError::ExceedsPermit,
        }
    }
}

impl ExactHostAllocationPermit for HostTraceScratchPermit {
    fn reserve_exact(
        &self,
        _bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    }

    fn reserve_exact_claim(
        &self,
        bytes: usize,
    ) -> Result<ExactHostAllocationClaim, ExactHostNumericAllocationError> {
        self.check_local()
            .map_err(|error| self.save_failure(error))?;
        let claim = self
            .ledger
            .reserve_dynamic_scratch_claim(bytes, self.epoch)
            .map_err(|error| self.save_failure(HostAllocationError::Resource(error)))?;
        self.check_local()
            .map_err(|error| self.save_failure(error))?;
        Ok(claim)
    }
}
