//! Non-owning stage observation for consumers that need liveness across awaits.

use super::*;

pub(super) struct StageReservationView {
    reservation: Weak<dyn HostEffectStageReservation>,
    active_epoch: Arc<AtomicBool>,
    cancellation: runmat_execution::cancellation::CancellationToken,
    session_generation_owner: Arc<AtomicU64>,
    session_generation: u64,
}

impl StageReservationView {
    pub(super) fn new(
        reservation: Weak<dyn HostEffectStageReservation>,
        active_epoch: Arc<AtomicBool>,
        cancellation: runmat_execution::cancellation::CancellationToken,
        session_generation_owner: Arc<AtomicU64>,
        session_generation: u64,
    ) -> Self {
        Self {
            reservation,
            active_epoch,
            cancellation,
            session_generation_owner,
            session_generation,
        }
    }

    fn current_reservation(&self) -> Option<Arc<dyn HostEffectStageReservation>> {
        (!self.cancellation.is_cancelled()
            && self.active_epoch.load(Ordering::Acquire)
            && self.session_generation_owner.load(Ordering::Acquire) == self.session_generation)
            .then(|| self.reservation.upgrade())
            .flatten()
            .filter(|owner| owner.is_current())
    }
}

impl HostEffectStageReservation for StageReservationView {
    fn managed_ledger(&self) -> Option<super::super::ManagedHostEffectLedger> {
        self.current_reservation()?.managed_ledger()
    }

    fn reserved_bytes(&self) -> u64 {
        self.current_reservation()
            .map_or(0, |owner| owner.reserved_bytes())
    }

    fn capacity_bytes(&self) -> u64 {
        self.current_reservation()
            .map_or(0, |owner| owner.capacity_bytes())
    }

    fn live_claim_metadata_bytes(&self) -> usize {
        self.current_reservation()
            .map_or(0, |owner| owner.live_claim_metadata_bytes())
    }

    fn owned_output_claim_metadata_bytes(&self) -> usize {
        self.current_reservation()
            .map_or(0, |owner| owner.owned_output_claim_metadata_bytes())
    }

    fn reserve_live_exact(
        &self,
        kind: HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<Arc<dyn HostEffectLiveAllocationReservation>, HostEffectAllocationError> {
        if !self.active_epoch.load(Ordering::Acquire) {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        if self.cancellation.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        let owner = self
            .current_reservation()
            .ok_or(HostEffectAllocationError::Revoked)?;
        let claim = owner.reserve_live_exact(kind, bytes)?;
        if !self.active_epoch.load(Ordering::Acquire) {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        if self.cancellation.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if self.session_generation_owner.load(Ordering::Acquire) != self.session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        Ok(claim)
    }

    fn reserve_output_headroom(
        &self,
        _bytes: u64,
    ) -> Result<Arc<dyn HostEffectOutputHeadroom>, HostEffectAllocationError> {
        // This view exists for retained scan liveness. Only the receipt's
        // bounded output permit may reserve a physically retained result.
        Err(HostEffectAllocationError::Unavailable)
    }

    fn is_current(&self) -> bool {
        self.current_reservation().is_some()
    }
}
