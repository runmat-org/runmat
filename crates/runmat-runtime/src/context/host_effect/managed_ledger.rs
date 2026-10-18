//! Reusable host-owned byte accounting, not a workload resource certificate.
//! A host adapter must still certify and install its complete-call bounds.

use super::{
    HostEffectAllocationError, HostEffectLiveAllocationClaim, HostEffectLiveAllocationKind,
    HostEffectLiveAllocationReservation, HostEffectOutputHeadroom, HostEffectStageReservation,
};
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationReservation, ExactHostNumericAllocationError,
};
use std::fmt;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, Weak,
};

mod pool;
pub use pool::{ManagedHostEffectStagePool, ManagedHostEffectStageSlot};
mod output_claim;
use output_claim::{OwnedOutputClaim, OwnedOutputClaimRefund};

pub(crate) const fn owned_output_claim_metadata_bytes() -> usize {
    OwnedOutputClaim::metadata_bytes()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManagedHostEffectLedgerSnapshot {
    /// Immutable issuer epoch. Rebinding to a different epoch revokes new
    /// claims on the previous owner without releasing retained outputs.
    pub epoch: u64,
    pub capacity_bytes: u64,
    /// Conservative precharges and retained outputs, not process RSS.
    pub live_reserved_bytes: u64,
    pub peak_reserved_bytes: u64,
    /// Exact allocations actually claimed through this API.
    pub live_claimed_bytes: u64,
    pub peak_claimed_bytes: u64,
    pub current: bool,
}

impl ManagedHostEffectLedgerSnapshot {
    /// Free bytes are derived from the same atomic snapshot as live claims;
    /// callers must not subtract staged claims from a second budget snapshot.
    pub fn free_bytes(self) -> u64 {
        self.capacity_bytes - self.live_reserved_bytes
    }
}

#[derive(Debug)]
struct LedgerState {
    snapshot: ManagedHostEffectLedgerSnapshot,
}

impl OwnedOutputClaimRefund for Mutex<LedgerState> {
    fn refund_exact_output_claim(&self, charged: u64) {
        let mut state = self.lock().expect("host ledger lock");
        state.snapshot.live_claimed_bytes -= charged;
        state.snapshot.live_reserved_bytes -= charged;
    }
}

#[derive(Clone)]
pub struct ManagedHostEffectLedger {
    state: Arc<Mutex<LedgerState>>,
}

impl ManagedHostEffectLedger {
    /// Issue one dynamically sized, exact temporary backing. Unlike a stage
    /// precharge, this claim competes for live capacity at each actual growth;
    /// the owned claim header is freed before its charge is refunded.
    pub(crate) fn reserve_dynamic_scratch_claim(
        &self,
        bytes: usize,
        epoch: u64,
    ) -> Result<ExactHostAllocationClaim, HostEffectAllocationError> {
        let charged = bytes
            .checked_add(OwnedOutputClaim::metadata_bytes())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(HostEffectAllocationError::Overflow)?;
        let mut state = self.state.lock().expect("host ledger lock");
        if !state.snapshot.current || state.snapshot.epoch != epoch {
            return Err(HostEffectAllocationError::Revoked);
        }
        let reserved = state
            .snapshot
            .live_reserved_bytes
            .checked_add(charged)
            .ok_or(HostEffectAllocationError::Overflow)?;
        if reserved > state.snapshot.capacity_bytes {
            return Err(HostEffectAllocationError::ExceedsCapacity);
        }
        let claimed = state
            .snapshot
            .live_claimed_bytes
            .checked_add(charged)
            .ok_or(HostEffectAllocationError::Overflow)?;
        state.snapshot.live_reserved_bytes = reserved;
        state.snapshot.peak_reserved_bytes = state.snapshot.peak_reserved_bytes.max(reserved);
        state.snapshot.live_claimed_bytes = claimed;
        state.snapshot.peak_claimed_bytes = state.snapshot.peak_claimed_bytes.max(claimed);
        drop(state);
        let owner: Arc<dyn OwnedOutputClaimRefund> = self.state.clone();
        OwnedOutputClaim::allocate(owner, charged)
            .map_err(|_| HostEffectAllocationError::AllocationFailed)
    }

    /// Exact retained metadata outside a live invocation headroom, including
    /// a session-owned state or a view derived from an already published value.
    /// It competes with every stage/output reservation on the same ledger;
    /// neither a previous claim nor a completed invocation mints capacity.
    pub(crate) fn reserve_retained_allocation(
        &self,
        bytes: u64,
        epoch: u64,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, HostEffectAllocationError> {
        let mut ledger = self.state.lock().expect("host ledger lock");
        if !ledger.snapshot.current || ledger.snapshot.epoch != epoch {
            return Err(HostEffectAllocationError::Revoked);
        }
        let next = ledger
            .snapshot
            .live_reserved_bytes
            .checked_add(bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        if next > ledger.snapshot.capacity_bytes {
            return Err(HostEffectAllocationError::ExceedsCapacity);
        }
        let claimed = ledger
            .snapshot
            .live_claimed_bytes
            .checked_add(bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        ledger.snapshot.live_reserved_bytes = next;
        ledger.snapshot.peak_reserved_bytes = ledger.snapshot.peak_reserved_bytes.max(next);
        ledger.snapshot.live_claimed_bytes = claimed;
        ledger.snapshot.peak_claimed_bytes = ledger
            .snapshot
            .peak_claimed_bytes
            .max(ledger.snapshot.live_claimed_bytes);
        drop(ledger);
        Ok(Arc::new(OutputClaim {
            ledger: self.state.clone(),
            headroom: std::sync::Weak::new(),
            bytes,
        }))
    }
    pub fn new(capacity_bytes: u64) -> Self {
        Self::new_with_epoch(capacity_bytes, 0)
    }

    pub fn new_with_epoch(capacity_bytes: u64, epoch: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(LedgerState {
                snapshot: ManagedHostEffectLedgerSnapshot {
                    epoch,
                    capacity_bytes,
                    live_reserved_bytes: 0,
                    peak_reserved_bytes: 0,
                    live_claimed_bytes: 0,
                    peak_claimed_bytes: 0,
                    current: true,
                },
            })),
        }
    }

    pub fn snapshot(&self) -> ManagedHostEffectLedgerSnapshot {
        self.state.lock().expect("host ledger lock").snapshot
    }

    #[cfg(test)]
    pub(crate) fn set_epoch_for_test(&self, epoch: u64) {
        self.state.lock().expect("host ledger lock").snapshot.epoch = epoch;
    }

    pub fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    /// Revocation forbids new claims. It does not erase bytes held by owners.
    pub fn revoke(&self) {
        self.state
            .lock()
            .expect("host ledger lock")
            .snapshot
            .current = false;
    }

    pub fn reserve_stage(
        &self,
        reserved_bytes: u64,
    ) -> Result<Arc<ManagedHostEffectStageReservation>, HostEffectAllocationError> {
        let mut state = self.state.lock().expect("host ledger lock");
        if !state.snapshot.current {
            return Err(HostEffectAllocationError::Revoked);
        }
        let next = state
            .snapshot
            .live_reserved_bytes
            .checked_add(reserved_bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        if next > state.snapshot.capacity_bytes {
            return Err(HostEffectAllocationError::ExceedsCapacity);
        }
        state.snapshot.live_reserved_bytes = next;
        state.snapshot.peak_reserved_bytes = state.snapshot.peak_reserved_bytes.max(next);
        drop(state);
        Ok(Arc::new(ManagedHostEffectStageReservation {
            inner: Arc::new(StageInner {
                ledger: self.state.clone(),
                reserved_bytes: AtomicU64::new(reserved_bytes),
                temporary: Mutex::new(TemporaryState {
                    live_bytes: 0,
                    retire_pending: false,
                }),
            }),
        }))
    }
}

struct TemporaryState {
    live_bytes: u64,
    retire_pending: bool,
}

struct StageInner {
    ledger: Arc<Mutex<LedgerState>>,
    reserved_bytes: AtomicU64,
    temporary: Mutex<TemporaryState>,
}

impl Drop for StageInner {
    fn drop(&mut self) {
        let mut state = self.ledger.lock().expect("host ledger lock");
        state.snapshot.live_reserved_bytes -= self.reserved_bytes.load(Ordering::Acquire);
    }
}

#[derive(Clone)]
pub struct ManagedHostEffectStageReservation {
    inner: Arc<StageInner>,
}

impl ManagedHostEffectStageReservation {
    fn transfer_to_headroom(
        &self,
        headroom: &HeadroomInner,
        bytes: u64,
        epoch: u64,
    ) -> Result<(), HostEffectAllocationError> {
        if !Arc::ptr_eq(&self.inner.ledger, &headroom.ledger)
            || !Weak::ptr_eq(&Arc::downgrade(&self.inner), &headroom.stage)
        {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        let temporary = self.inner.temporary.lock().expect("temporary ledger lock");
        let mut output = headroom.state.lock().expect("output headroom lock");
        let ledger = self.inner.ledger.lock().expect("host ledger lock");
        if !ledger.snapshot.current || ledger.snapshot.epoch != epoch {
            return Err(HostEffectAllocationError::Revoked);
        }
        if !output.active || temporary.retire_pending {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        let reserved = self.inner.reserved_bytes.load(Ordering::Acquire);
        let remaining = reserved
            .checked_sub(bytes)
            .ok_or(HostEffectAllocationError::ExceedsStageEnvelope)?;
        if remaining < temporary.live_bytes {
            return Err(HostEffectAllocationError::ExceedsStageEnvelope);
        }
        let next_output = output
            .unallocated
            .checked_add(bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        // The global reservation is unchanged: both shares belong to the
        // same ledger, and all checks above precede either mutation.
        self.inner
            .reserved_bytes
            .store(remaining, Ordering::Release);
        output.unallocated = next_output;
        Ok(())
    }

    /// Exact metadata retained by one owned temporary claim. Ordinary
    /// numeric workspaces use the same stage claim and must include this in
    /// their checked phase peak.
    pub fn exact_temporary_claim_metadata_bytes() -> usize {
        std::mem::size_of::<TemporaryClaim>()
    }

    fn claim_temporary(
        &self,
        bytes: usize,
        metadata_bytes: usize,
    ) -> Result<TemporaryClaim, HostEffectAllocationError> {
        let bytes = bytes
            .checked_add(metadata_bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        let bytes = u64::try_from(bytes).map_err(|_| HostEffectAllocationError::Overflow)?;
        let mut temporary = self.inner.temporary.lock().expect("temporary ledger lock");
        let mut ledger = self.inner.ledger.lock().expect("host ledger lock");
        if !ledger.snapshot.current {
            return Err(HostEffectAllocationError::Revoked);
        }
        let next = temporary
            .live_bytes
            .checked_add(bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        if next > self.inner.reserved_bytes.load(Ordering::Acquire) {
            return Err(HostEffectAllocationError::ExceedsStageEnvelope);
        }
        let claimed = ledger
            .snapshot
            .live_claimed_bytes
            .checked_add(bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        temporary.live_bytes = next;
        ledger.snapshot.live_claimed_bytes = claimed;
        ledger.snapshot.peak_claimed_bytes = ledger.snapshot.peak_claimed_bytes.max(claimed);
        Ok(TemporaryClaim {
            stage: self.inner.clone(),
            bytes,
        })
    }

    fn retire_pooled_invocation(&self) {
        let mut temporary = self.inner.temporary.lock().expect("temporary ledger lock");
        let mut ledger = self.inner.ledger.lock().expect("host ledger lock");
        if temporary.live_bytes == 0 {
            let reserved = self.inner.reserved_bytes.swap(0, Ordering::AcqRel);
            ledger.snapshot.live_reserved_bytes -= reserved;
            temporary.retire_pending = false;
        } else {
            temporary.retire_pending = true;
        }
    }

    pub fn reserve_output_headroom(
        &self,
        bytes: u64,
    ) -> Result<Arc<dyn HostEffectOutputHeadroom>, HostEffectAllocationError> {
        let mut state = self.inner.ledger.lock().expect("host ledger lock");
        if !state.snapshot.current {
            return Err(HostEffectAllocationError::Revoked);
        }
        let reserved = state
            .snapshot
            .live_reserved_bytes
            .checked_add(bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        if reserved > state.snapshot.capacity_bytes {
            return Err(HostEffectAllocationError::ExceedsCapacity);
        }
        state.snapshot.live_reserved_bytes = reserved;
        state.snapshot.peak_reserved_bytes = state.snapshot.peak_reserved_bytes.max(reserved);
        Ok(Arc::new(ManagedOutputHeadroom {
            inner: Arc::new(HeadroomInner {
                ledger: self.inner.ledger.clone(),
                stage: Arc::downgrade(&self.inner),
                state: Mutex::new(HeadroomState {
                    unallocated: bytes,
                    active: true,
                }),
            }),
        }))
    }
}

impl HostEffectStageReservation for ManagedHostEffectStageReservation {
    fn managed_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(ManagedHostEffectLedger {
            state: self.inner.ledger.clone(),
        })
    }

    fn reserved_bytes(&self) -> u64 {
        self.inner.reserved_bytes.load(Ordering::Acquire)
    }

    fn promote_scan_envelope(
        &self,
        new_reserved_bytes: u64,
    ) -> Result<(), HostEffectAllocationError> {
        // The same lock order as temporary claims keeps their envelope and
        // the global precharge in one observable state.
        let temporary = self.inner.temporary.lock().expect("temporary ledger lock");
        let mut ledger = self.inner.ledger.lock().expect("host ledger lock");
        if !ledger.snapshot.current {
            return Err(HostEffectAllocationError::Revoked);
        }
        let old = self.inner.reserved_bytes.load(Ordering::Acquire);
        if new_reserved_bytes < old || new_reserved_bytes < temporary.live_bytes {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        let delta = new_reserved_bytes - old;
        let next = ledger
            .snapshot
            .live_reserved_bytes
            .checked_add(delta)
            .ok_or(HostEffectAllocationError::Overflow)?;
        if next > ledger.snapshot.capacity_bytes {
            return Err(HostEffectAllocationError::ExceedsCapacity);
        }
        ledger.snapshot.live_reserved_bytes = next;
        ledger.snapshot.peak_reserved_bytes = ledger.snapshot.peak_reserved_bytes.max(next);
        self.inner
            .reserved_bytes
            .store(new_reserved_bytes, Ordering::Release);
        Ok(())
    }

    fn capacity_bytes(&self) -> u64 {
        self.inner
            .ledger
            .lock()
            .expect("host ledger lock")
            .snapshot
            .capacity_bytes
    }

    fn reserve_live_exact(
        &self,
        _kind: HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<Arc<dyn HostEffectLiveAllocationReservation>, HostEffectAllocationError> {
        Ok(Arc::new(self.claim_temporary(bytes, 0)?))
    }

    fn reserve_live_exact_claim(
        &self,
        _kind: HostEffectLiveAllocationKind,
        bytes: usize,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
        let claim = self.claim_temporary(bytes, std::mem::size_of::<TemporaryClaim>())?;
        let layout = std::alloc::Layout::new::<TemporaryClaim>();
        // SAFETY: the allocation has the exact nonzero claim layout. On
        // failure `claim` drops and refunds the temporary ledger charge.
        let pointer = unsafe { std::alloc::alloc(layout) }.cast::<TemporaryClaim>();
        if pointer.is_null() {
            return Err(HostEffectAllocationError::AllocationFailed);
        }
        // SAFETY: this fresh allocation is initialized exactly once and
        // handed to the sole Box owner without an intervening fallible step.
        unsafe {
            pointer.write(claim);
            Ok(HostEffectLiveAllocationClaim::Owned(
                runmat_async::ManagedDiagnosticAllocation::new(Box::from_raw(pointer)),
            ))
        }
    }

    fn live_claim_metadata_bytes(&self) -> usize {
        std::mem::size_of::<TemporaryClaim>()
    }

    fn reserve_output_headroom(
        &self,
        bytes: u64,
    ) -> Result<Arc<dyn HostEffectOutputHeadroom>, HostEffectAllocationError> {
        ManagedHostEffectStageReservation::reserve_output_headroom(self, bytes)
    }

    fn is_current(&self) -> bool {
        self.inner
            .ledger
            .lock()
            .expect("host ledger lock")
            .snapshot
            .current
    }
}

struct TemporaryClaim {
    stage: Arc<StageInner>,
    bytes: u64,
}

impl fmt::Debug for TemporaryClaim {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TemporaryClaim")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl runmat_async::ManagedDiagnosticAllocationLease for TemporaryClaim {
    fn retire_owned(self: Box<Self>) {
        // SAFETY: the unique Box came from the global allocator with the
        // exact TemporaryClaim layout and is retired exactly once.
        unsafe {
            Self::retire_owned_with(self, |pointer, layout| {
                std::alloc::dealloc(pointer.as_ptr(), layout)
            })
        }
    }
}

impl HostEffectLiveAllocationReservation for TemporaryClaim {}

impl TemporaryClaim {
    /// Release the claim header before refunding the bytes that funded it.
    ///
    /// # Safety
    /// `deallocate` must release the supplied exact global-allocator block
    /// once and must not unwind after ownership leaves its Box.
    unsafe fn retire_owned_with(
        claim: Box<Self>,
        deallocate: impl FnOnce(std::ptr::NonNull<u8>, std::alloc::Layout),
    ) {
        let pointer = Box::into_raw(claim);
        // SAFETY: consuming the unique Box makes the stage field uniquely
        // owned until it is dropped after the refund below.
        let stage = unsafe { std::ptr::read(std::ptr::addr_of!((*pointer).stage)) };
        // SAFETY: the allocation remains live through this scalar read.
        let bytes = unsafe { (*pointer).bytes };
        deallocate(
            unsafe { std::ptr::NonNull::new_unchecked(pointer.cast::<u8>()) },
            std::alloc::Layout::new::<Self>(),
        );
        Self::refund(&stage, bytes);
    }

    fn refund(stage: &StageInner, bytes: u64) {
        let mut temporary = stage.temporary.lock().expect("temporary ledger lock");
        let mut ledger = stage.ledger.lock().expect("host ledger lock");
        temporary.live_bytes -= bytes;
        ledger.snapshot.live_claimed_bytes -= bytes;
        if temporary.live_bytes == 0 && temporary.retire_pending {
            let reserved = stage.reserved_bytes.swap(0, Ordering::AcqRel);
            ledger.snapshot.live_reserved_bytes -= reserved;
            temporary.retire_pending = false;
        }
    }
}

impl Drop for TemporaryClaim {
    fn drop(&mut self) {
        Self::refund(&self.stage, self.bytes);
    }
}

struct OutputClaim {
    ledger: Arc<Mutex<LedgerState>>,
    headroom: std::sync::Weak<HeadroomInner>,
    bytes: u64,
}

pub(crate) fn retained_claim_host_upper_bytes() -> u64 {
    // Arc's strong/weak counters and alignment accompany the claim object.
    u64::try_from(
        std::mem::size_of::<OutputClaim>()
            + 2 * std::mem::size_of::<usize>()
            + std::mem::align_of::<OutputClaim>(),
    )
    .expect("claim metadata fits u64")
}

struct ManagedOutputHeadroom {
    inner: Arc<HeadroomInner>,
}

struct HeadroomInner {
    ledger: Arc<Mutex<LedgerState>>,
    stage: Weak<StageInner>,
    state: Mutex<HeadroomState>,
}

impl fmt::Debug for HeadroomInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HeadroomInner")
            .finish_non_exhaustive()
    }
}

struct HeadroomState {
    unallocated: u64,
    active: bool,
}

impl fmt::Debug for ManagedOutputHeadroom {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedOutputHeadroom")
            .finish_non_exhaustive()
    }
}

impl HostEffectOutputHeadroom for ManagedOutputHeadroom {
    fn transfer_settled_stage_bytes(
        &self,
        stage: &ManagedHostEffectStageReservation,
        bytes: u64,
        epoch: u64,
    ) -> Result<(), HostEffectAllocationError> {
        stage.transfer_to_headroom(&self.inner, bytes, epoch)
    }

    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        let bytes =
            u64::try_from(bytes).map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        let mut headroom = self.inner.state.lock().expect("output headroom lock");
        if !headroom.active || bytes > headroom.unallocated {
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        let mut ledger = self.inner.ledger.lock().expect("host ledger lock");
        if !ledger.snapshot.current {
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        let claimed = ledger
            .snapshot
            .live_claimed_bytes
            .checked_add(bytes)
            .ok_or(ExactHostNumericAllocationError::ExceedsPermit)?;
        headroom.unallocated -= bytes;
        ledger.snapshot.live_claimed_bytes = claimed;
        ledger.snapshot.peak_claimed_bytes = ledger.snapshot.peak_claimed_bytes.max(claimed);
        Ok(Arc::new(OutputClaim {
            ledger: self.inner.ledger.clone(),
            headroom: Arc::downgrade(&self.inner),
            bytes,
        }))
    }

    fn reserve_exact_claim(
        &self,
        bytes: usize,
    ) -> Result<runmat_value::ExactHostAllocationClaim, ExactHostNumericAllocationError> {
        let metadata = OwnedOutputClaim::metadata_bytes();
        let charged = bytes
            .checked_add(metadata)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(ExactHostNumericAllocationError::ExceedsPermit)?;
        let mut headroom = self.inner.state.lock().expect("output headroom lock");
        if !headroom.active || charged > headroom.unallocated {
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        let mut ledger = self.inner.ledger.lock().expect("host ledger lock");
        if !ledger.snapshot.current {
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        let claimed = ledger
            .snapshot
            .live_claimed_bytes
            .checked_add(charged)
            .ok_or(ExactHostNumericAllocationError::ExceedsPermit)?;
        headroom.unallocated -= charged;
        ledger.snapshot.live_claimed_bytes = claimed;
        ledger.snapshot.peak_claimed_bytes = ledger.snapshot.peak_claimed_bytes.max(claimed);
        drop(ledger);
        drop(headroom);
        let owner: Arc<dyn OwnedOutputClaimRefund> = self.inner.clone();
        OwnedOutputClaim::allocate(owner, charged)
    }

    fn retire_invocation(&self) {
        let mut headroom = self.inner.state.lock().expect("output headroom lock");
        if !headroom.active {
            return;
        }
        let mut ledger = self.inner.ledger.lock().expect("host ledger lock");
        ledger.snapshot.live_reserved_bytes -= headroom.unallocated;
        headroom.unallocated = 0;
        headroom.active = false;
    }
}

impl OwnedOutputClaimRefund for HeadroomInner {
    fn refund_exact_output_claim(&self, charged: u64) {
        let mut headroom = self.state.lock().expect("output headroom lock");
        let mut state = self.ledger.lock().expect("host ledger lock");
        if headroom.active {
            headroom.unallocated += charged;
        } else {
            state.snapshot.live_reserved_bytes -= charged;
        }
        state.snapshot.live_claimed_bytes -= charged;
    }
}

impl Drop for HeadroomInner {
    fn drop(&mut self) {
        let remaining = self.state.lock().expect("output headroom lock").unallocated;
        self.ledger
            .lock()
            .expect("host ledger lock")
            .snapshot
            .live_reserved_bytes -= remaining;
    }
}

impl fmt::Debug for OutputClaim {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OutputClaim")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl ExactHostAllocationReservation for OutputClaim {}

impl Drop for OutputClaim {
    fn drop(&mut self) {
        if let Some(headroom) = self.headroom.upgrade() {
            let mut headroom = headroom.state.lock().expect("output headroom lock");
            let mut state = self.ledger.lock().expect("host ledger lock");
            if headroom.active {
                headroom.unallocated += self.bytes;
            } else {
                state.snapshot.live_reserved_bytes -= self.bytes;
            }
            state.snapshot.live_claimed_bytes -= self.bytes;
        } else {
            let mut state = self.ledger.lock().expect("host ledger lock");
            state.snapshot.live_reserved_bytes -= self.bytes;
            state.snapshot.live_claimed_bytes -= self.bytes;
        }
    }
}

#[cfg(test)]
#[path = "managed_ledger/tests.rs"]
mod tests;
