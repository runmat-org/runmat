//! Dynamically admitted, exact physical traversal scratch for Value tracing.

use std::fmt;
use std::sync::{Arc, Mutex};

use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostAllocationReservation,
    ExactHostNumericAllocationError,
};

use super::HostAllocationError;
use crate::context::host_effect::HostEffectAllocationError;
use crate::context::RuntimeContext;

/// One operation's issuer, not a reservation. Each growth reserves the exact
/// frame backing plus the canonical owned-claim header before allocating it.
pub(crate) struct HostTraceScratchPermit {
    owner: super::HostControlAllocationOwner,
    last_failure: Mutex<Option<HostAllocationError>>,
}

impl fmt::Debug for HostTraceScratchPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HostTraceScratchPermit")
            .field("owner", &self.owner)
            .finish_non_exhaustive()
    }
}

impl HostTraceScratchPermit {
    pub(super) fn new(owner: super::HostControlAllocationOwner) -> Self {
        Self {
            owner,
            last_failure: Mutex::new(None),
        }
    }

    pub(crate) fn checkpoint(&self, runtime: &RuntimeContext) -> Result<(), HostAllocationError> {
        self.owner.checkpoint(runtime)
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
        self.owner
            .claim(bytes)
            .map_err(|error| self.save_failure(error))
    }
}
