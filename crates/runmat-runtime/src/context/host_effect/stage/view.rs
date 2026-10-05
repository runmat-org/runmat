//! Non-owning stage observation for consumers that need liveness across awaits.

use super::*;

pub(super) struct StageReservationView {
    reservation: Weak<dyn HostEffectStageReservation>,
    active_epoch: Arc<AtomicBool>,
    cancellation: runmat_execution::cancellation::CancellationToken,
}

impl StageReservationView {
    pub(super) fn new(
        reservation: Weak<dyn HostEffectStageReservation>,
        active_epoch: Arc<AtomicBool>,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> Self {
        Self {
            reservation,
            active_epoch,
            cancellation,
        }
    }

    fn current_reservation(&self) -> Option<Arc<dyn HostEffectStageReservation>> {
        (!self.cancellation.is_cancelled() && self.active_epoch.load(Ordering::Acquire))
            .then(|| self.reservation.upgrade())
            .flatten()
            .filter(|owner| owner.is_current())
    }
}

impl HostEffectStageReservation for StageReservationView {
    fn reserved_bytes(&self) -> u64 {
        self.current_reservation()
            .map_or(0, |owner| owner.reserved_bytes())
    }

    fn capacity_bytes(&self) -> u64 {
        self.current_reservation()
            .map_or(0, |owner| owner.capacity_bytes())
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
        Ok(claim)
    }

    fn reserve_output_exact(
        &self,
        _bytes: usize,
    ) -> Result<
        Arc<dyn runmat_value::ExactHostAllocationReservation>,
        runmat_value::ExactHostNumericAllocationError,
    > {
        // This view exists for retained scan liveness. Only the receipt's
        // bounded output permit may reserve a physically retained result.
        Err(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
    }

    fn is_current(&self) -> bool {
        self.current_reservation().is_some()
    }
}
