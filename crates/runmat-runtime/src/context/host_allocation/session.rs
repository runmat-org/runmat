//! Retained session metadata shares the ordinary managed host ledger.
//!
//! A session may be constructed before a finite parallel service is installed.
//! Domains therefore claim a retained payload at its first materialization,
//! and rehome it under a different owner before creating another allocation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use runmat_value::ExactHostAllocationReservation;

use super::{HostAllocationError, HostAllocationPlan};
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::RuntimeContext;

pub(crate) struct HostRetainedSessionAllocation {
    ledger: ManagedHostEffectLedger,
    epoch: u64,
    session_identity: crate::context::RuntimeContextLocalId,
    generation_owner: Arc<AtomicU64>,
    generation: u64,
    _claim: Arc<dyn ExactHostAllocationReservation>,
}

impl std::fmt::Debug for HostRetainedSessionAllocation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostRetainedSessionAllocation")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl HostRetainedSessionAllocation {
    pub(crate) fn admit(
        runtime: Option<&RuntimeContext>,
        payload_bytes: u64,
    ) -> Result<Option<Self>, HostAllocationError> {
        let Some(runtime) = runtime else {
            return Ok(None);
        };
        if runtime.is_cancelled() {
            return Err(HostAllocationError::Cancelled);
        }
        let Some(parallel) = runtime.service_ports().parallel() else {
            return Ok(None);
        };
        let resources = parallel.placement_resources();
        if resources.memory_available_bytes.is_none() {
            return Ok(None);
        }
        let ledger = parallel
            .managed_host_effect_ledger()
            .ok_or(HostAllocationError::MissingManagedOwner)?;
        let snapshot = ledger.snapshot();
        if !snapshot.current || snapshot.epoch != resources.epoch {
            return Err(HostAllocationError::StaleAuthority);
        }
        let bytes = HostAllocationPlan::retained_metadata_bytes(payload_bytes)?;
        let (generation_owner, generation) = runtime.host_effect_session_generation();
        let claim = ledger.reserve_retained_allocation(bytes, resources.epoch)?;
        let allocation = Self {
            ledger,
            epoch: resources.epoch,
            session_identity: runtime.local_identity(),
            generation_owner,
            generation,
            _claim: claim,
        };
        if runtime.is_cancelled() {
            return Err(HostAllocationError::Cancelled);
        }
        if !allocation.matches_current_owner(runtime) {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(Some(allocation))
    }

    pub(crate) fn matches_current_owner(&self, runtime: &RuntimeContext) -> bool {
        !runtime.is_cancelled()
            && self.generation_owner.load(Ordering::Acquire) == self.generation
            && runtime.local_identity() == self.session_identity
            && runtime
                .service_ports()
                .parallel()
                .and_then(|parallel| {
                    let resources = parallel.placement_resources();
                    parallel
                        .managed_host_effect_ledger()
                        .map(|ledger| (resources, ledger))
                })
                .is_some_and(|(resources, ledger)| {
                    let snapshot = ledger.snapshot();
                    resources.memory_available_bytes.is_some()
                        && resources.epoch == self.epoch
                        && ledger.same_owner(&self.ledger)
                        && snapshot.current
                        && snapshot.epoch == self.epoch
                })
    }
}
