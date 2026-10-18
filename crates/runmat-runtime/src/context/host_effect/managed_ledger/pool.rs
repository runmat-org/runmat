//! Setup-owned reusable stage reservations. The pool allocates its Arc slots
//! before any invocation; an acquired call only clones an existing Arc and
//! takes a checked generation. It does not certify a workload peak.

use super::*;
use std::sync::atomic::AtomicBool;
mod output;
use output::PooledOutputPermit;

pub struct ManagedHostEffectStagePool {
    slots: Vec<Arc<PooledStageSlot>>,
    admission: Mutex<()>,
}

pub struct ManagedHostEffectStageSlot {
    reservation: Option<Arc<dyn HostEffectStageReservation>>,
    pub generation: u64,
}

impl ManagedHostEffectStageSlot {
    pub fn reservation(&self) -> &Arc<dyn HostEffectStageReservation> {
        self.reservation.as_ref().expect("live stage acquisition")
    }

    pub fn into_reservation(mut self) -> Arc<dyn HostEffectStageReservation> {
        self.reservation.take().expect("live stage acquisition")
    }
}

impl Drop for ManagedHostEffectStageSlot {
    fn drop(&mut self) {
        if let Some(reservation) = &self.reservation {
            reservation.retire_invocation();
        }
    }
}

struct PooledStageSlot {
    reservation: ManagedHostEffectStageReservation,
    output: Arc<PooledOutputPermit>,
    mutation: Mutex<()>,
    live_reservation: AtomicBool,
    active_epoch: Arc<AtomicBool>,
    generation: AtomicU64,
    retired: AtomicBool,
}

impl ManagedHostEffectStagePool {
    pub const fn owned_output_claim_metadata_bytes() -> usize {
        super::OwnedOutputClaim::metadata_bytes()
    }

    pub fn new(
        ledger: &ManagedHostEffectLedger,
        capacity: usize,
    ) -> Result<Self, HostEffectAllocationError> {
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(capacity)
            .map_err(|_| HostEffectAllocationError::AllocationFailed)?;
        for _ in 0..capacity {
            let reservation = ledger.reserve_stage(0)?;
            let active_epoch = Arc::new(AtomicBool::new(false));
            slots.push(Arc::new(PooledStageSlot {
                reservation: Arc::try_unwrap(reservation)
                    .unwrap_or_else(|_| unreachable!("new zero-byte reservation has one owner")),
                output: PooledOutputPermit::new(ledger, active_epoch.clone()),
                mutation: Mutex::new(()),
                live_reservation: AtomicBool::new(false),
                active_epoch,
                generation: AtomicU64::new(0),
                retired: AtomicBool::new(false),
            }));
        }
        Ok(Self {
            slots,
            admission: Mutex::new(()),
        })
    }

    pub fn acquire(
        &self,
        bytes: u64,
    ) -> Result<ManagedHostEffectStageSlot, HostEffectAllocationError> {
        let _admission = self.admission.lock().expect("managed stage pool lock");
        for slot in &self.slots {
            let _mutation = slot.mutation.lock().expect("managed stage slot lock");
            if slot.live_reservation.load(Ordering::Acquire)
                || slot.retired.load(Ordering::Acquire)
                || Arc::strong_count(slot) != 1
                || Arc::weak_count(slot) != 0
                || Arc::strong_count(&slot.active_epoch) != 2
                || Arc::strong_count(&slot.output) != 1
                || slot.reservation.reserved_bytes() != 0
            {
                continue;
            }
            let Some(generation) = slot.generation.load(Ordering::Acquire).checked_add(1) else {
                slot.retired.store(true, Ordering::Release);
                continue;
            };
            slot.reservation.promote_scan_envelope(bytes)?;
            slot.generation.store(generation, Ordering::Release);
            slot.live_reservation.store(true, Ordering::Release);
            slot.active_epoch.store(true, Ordering::Release);
            return Ok(ManagedHostEffectStageSlot {
                reservation: Some(Arc::clone(slot) as Arc<dyn HostEffectStageReservation>),
                generation,
            });
        }
        Err(HostEffectAllocationError::ExceedsCapacity)
    }
}

impl HostEffectStageReservation for PooledStageSlot {
    fn transfer_settled_stage_bytes(
        &self,
        bytes: u64,
        epoch: u64,
    ) -> Result<(), HostEffectAllocationError> {
        let _mutation = self.mutation.lock().expect("managed stage slot lock");
        if !self.live_reservation.load(Ordering::Acquire)
            || !self.active_epoch.load(Ordering::Acquire)
        {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        self.output
            .transfer_settled_stage_bytes(&self.reservation, bytes, epoch)
    }

    fn managed_ledger(&self) -> Option<ManagedHostEffectLedger> {
        self.reservation.managed_ledger()
    }

    fn reserved_bytes(&self) -> u64 {
        self.reservation.reserved_bytes()
    }

    fn capacity_bytes(&self) -> u64 {
        self.reservation.capacity_bytes()
    }

    fn promote_scan_envelope(&self, bytes: u64) -> Result<(), HostEffectAllocationError> {
        let _mutation = self.mutation.lock().expect("managed stage slot lock");
        if !self.live_reservation.load(Ordering::Acquire)
            || !self.active_epoch.load(Ordering::Acquire)
        {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        self.reservation.promote_scan_envelope(bytes)
    }

    fn reserve_live_exact(
        &self,
        kind: HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<Arc<dyn HostEffectLiveAllocationReservation>, HostEffectAllocationError> {
        let _mutation = self.mutation.lock().expect("managed stage slot lock");
        if !self.live_reservation.load(Ordering::Acquire)
            || !self.active_epoch.load(Ordering::Acquire)
        {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        self.reservation.reserve_live_exact(kind, bytes)
    }

    fn reserve_live_exact_claim(
        &self,
        kind: HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
        let _mutation = self.mutation.lock().expect("managed stage slot lock");
        if !self.live_reservation.load(Ordering::Acquire)
            || !self.active_epoch.load(Ordering::Acquire)
        {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        self.reservation.reserve_live_exact_claim(kind, bytes)
    }

    fn live_claim_metadata_bytes(&self) -> usize {
        self.reservation.live_claim_metadata_bytes()
    }

    fn reserve_output_headroom(
        &self,
        _bytes: u64,
    ) -> Result<Arc<dyn HostEffectOutputHeadroom>, HostEffectAllocationError> {
        // A dynamic headroom Arc would bypass the setup-owned exact slot.
        Err(HostEffectAllocationError::Unavailable)
    }

    fn reserve_preallocated_output_permit(
        &self,
        bytes: u64,
        cancellation: runmat_execution::cancellation::CancellationToken,
        session_generation_owner: Arc<AtomicU64>,
        session_generation: u64,
    ) -> Result<Option<Arc<dyn runmat_value::ExactHostAllocationPermit>>, HostEffectAllocationError>
    {
        let _mutation = self.mutation.lock().expect("managed stage slot lock");
        if !self.live_reservation.load(Ordering::Acquire)
            || !self.active_epoch.load(Ordering::Acquire)
        {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        self.output.prepare(
            bytes,
            cancellation,
            session_generation_owner,
            session_generation,
        )?;
        Ok(Some(self.output.clone()))
    }

    fn owned_output_claim_metadata_bytes(&self) -> usize {
        super::OwnedOutputClaim::metadata_bytes()
    }

    fn is_current(&self) -> bool {
        self.live_reservation.load(Ordering::Acquire)
            && self.active_epoch.load(Ordering::Acquire)
            && self.reservation.is_current()
    }

    fn slot_generation(&self) -> Option<u64> {
        Some(self.generation.load(Ordering::Acquire))
    }

    fn invocation_epoch(&self) -> Option<Arc<AtomicBool>> {
        Some(self.active_epoch.clone())
    }

    fn retire_invocation(&self) {
        let _mutation = self.mutation.lock().expect("managed stage slot lock");
        self.active_epoch.store(false, Ordering::Release);
        if self.live_reservation.swap(false, Ordering::AcqRel) {
            self.output.retire();
            self.reservation.retire_pooled_invocation();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::cancellation::CancellationSource;
    use std::sync::Barrier;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pooled_settled_stage_transfer_uses_the_existing_output_permit() {
        let bytes = 4 + ManagedHostEffectStagePool::owned_output_claim_metadata_bytes() as u64;
        let ledger = ManagedHostEffectLedger::new_with_epoch(bytes, 3);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let slot = pool.acquire(bytes).unwrap();
        let session = Arc::new(AtomicU64::new(1));
        let cancellation = CancellationSource::new();
        let permit = slot
            .reservation()
            .reserve_preallocated_output_permit(0, cancellation.token(), session, 1)
            .unwrap()
            .unwrap();
        slot.reservation()
            .transfer_settled_stage_bytes(bytes, 3)
            .expect("settled pooled stage");
        assert_eq!(slot.reservation().reserved_bytes(), 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, bytes);
        let claim = permit.reserve_exact_claim(4).expect("retained payload");
        assert_eq!(ledger.snapshot().peak_reserved_bytes, bytes);
        cancellation.cancel(runmat_execution::CancellationReason::User);
        assert_eq!(
            slot.reservation().transfer_settled_stage_bytes(1, 3),
            Err(HostEffectAllocationError::Cancelled)
        );
        slot.reservation().retire_invocation();
        drop(slot);
        drop(permit);
        assert_eq!(ledger.snapshot().live_reserved_bytes, bytes);
        drop(claim);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    #[test]
    fn slot_retires_once_and_reuse_requires_all_old_references_to_drop() {
        let ledger = ManagedHostEffectLedger::new(16);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let first = pool.acquire(8).unwrap();
        assert_eq!(first.generation, 1);
        assert_eq!(first.reservation().slot_generation(), Some(1));
        assert_eq!(ledger.snapshot().live_reserved_bytes, 8);
        let retained = first.reservation().clone();
        first.reservation().retire_invocation();
        first.reservation().retire_invocation();
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert!(matches!(
            pool.acquire(8),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        drop(first);
        assert!(matches!(
            pool.acquire(8),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        drop(retained);
        let next = pool.acquire(8).unwrap();
        assert_eq!(next.generation, 2);
        next.reservation().retire_invocation();
        drop(next);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn old_invocation_epoch_cannot_be_reactivated_by_slot_reuse() {
        let ledger = ManagedHostEffectLedger::new(16);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let first = pool.acquire(8).unwrap();
        let old_epoch = first.reservation().invocation_epoch().unwrap();
        assert!(old_epoch.load(Ordering::Acquire));
        first.reservation().retire_invocation();
        drop(first);
        assert!(!old_epoch.load(Ordering::Acquire));
        assert!(matches!(
            pool.acquire(8),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        drop(old_epoch);
        let second = pool.acquire(8).unwrap();
        assert_eq!(second.generation, 2);
        second.reservation().retire_invocation();
    }

    #[test]
    fn caller_epoch_closure_before_reservation_retirement_still_releases_bytes() {
        let ledger = ManagedHostEffectLedger::new(16);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let first = pool.acquire(8).unwrap();
        first
            .reservation()
            .invocation_epoch()
            .unwrap()
            .store(false, Ordering::Release);
        first.reservation().retire_invocation();
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        drop(first);
        assert_eq!(pool.acquire(8).unwrap().generation, 2);
    }

    #[test]
    fn temporary_claim_and_returned_output_outlive_distinct_stage_lifetimes() {
        let output_bytes =
            4 + ManagedHostEffectStagePool::owned_output_claim_metadata_bytes() as u64;
        let ledger = ManagedHostEffectLedger::new(8 + output_bytes);
        let pool = ManagedHostEffectStagePool::new(&ledger, 2).unwrap();
        let first = pool.acquire(8).unwrap();
        let scratch = first
            .reservation()
            .reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 4)
            .unwrap();
        let output_permit = first
            .reservation()
            .reserve_preallocated_output_permit(
                output_bytes,
                CancellationSource::new().token(),
                Arc::new(AtomicU64::new(1)),
                1,
            )
            .unwrap()
            .expect("setup-owned output permit");
        let output = output_permit.reserve_exact_claim(4).unwrap();
        first.reservation().retire_invocation();
        assert_eq!(ledger.snapshot().live_claimed_bytes, 4 + output_bytes);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 8 + output_bytes);
        drop(first);
        assert!(matches!(
            pool.acquire(4),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        drop(scratch);
        assert_eq!(ledger.snapshot().live_reserved_bytes, output_bytes);
        let second = pool
            .acquire(4)
            .expect("scratch retirement permits another preallocated stage slot");
        assert_eq!(second.generation, 1);
        assert_eq!(ledger.snapshot().live_reserved_bytes, output_bytes + 4);
        second.reservation().retire_invocation();
        drop(second);
        drop(output_permit);
        assert_eq!(ledger.snapshot().live_reserved_bytes, output_bytes);
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        let reused = pool
            .acquire(4)
            .expect("old output owner released its stage slot");
        assert_eq!(reused.generation, 2);
        drop(reused);
    }

    #[test]
    fn revoked_ledger_rejects_new_stage_without_reserving_bytes() {
        let ledger = ManagedHostEffectLedger::new(16);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        ledger.revoke();
        assert!(matches!(
            pool.acquire(8),
            Err(HostEffectAllocationError::Revoked)
        ));
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn untransferred_acquisition_retires_on_drop() {
        let ledger = ManagedHostEffectLedger::new(16);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        drop(pool.acquire(8).unwrap());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let next = pool.acquire(8).unwrap();
        assert_eq!(next.generation, 2);
    }

    #[test]
    fn distinct_pool_owners_share_capacity_without_cross_retirement() {
        let ledger = ManagedHostEffectLedger::new(16);
        let first_pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let second_pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let first = first_pool.acquire(8).unwrap();
        let second = second_pool.acquire(8).unwrap();
        assert_eq!(ledger.snapshot().live_reserved_bytes, 16);
        assert!(matches!(
            first_pool.acquire(1),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        first.reservation().retire_invocation();
        drop(first);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 8);
        assert!(second.reservation().is_current());
        second.reservation().retire_invocation();
        drop(second);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn concurrent_promotion_and_retirement_cannot_reserve_an_inactive_slot() {
        let ledger = ManagedHostEffectLedger::new(16);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        for _ in 0..64 {
            let slot = pool.acquire(8).unwrap();
            let promote = slot.reservation().clone();
            let retire = slot.reservation().clone();
            let start = Arc::new(Barrier::new(3));
            let promotion_start = start.clone();
            let promotion = std::thread::spawn(move || {
                promotion_start.wait();
                promote.promote_scan_envelope(12)
            });
            let retirement_start = start.clone();
            let retirement = std::thread::spawn(move || {
                retirement_start.wait();
                retire.retire_invocation();
            });
            start.wait();
            let _ = promotion.join().unwrap();
            retirement.join().unwrap();
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            assert!(matches!(
                slot.reservation().promote_scan_envelope(12),
                Err(HostEffectAllocationError::InvalidInvocation)
            ));
            drop(slot);
        }
    }
}
