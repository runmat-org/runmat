//! Reusable host-owned byte accounting, not a workload resource certificate.
//! A host adapter must still certify and install its complete-call bounds.

use super::{
    HostEffectAllocationError, HostEffectLiveAllocationClaim, HostEffectLiveAllocationKind,
    HostEffectLiveAllocationReservation, HostEffectOutputHeadroom, HostEffectStageReservation,
};
use runmat_value::{ExactHostAllocationReservation, ExactHostNumericAllocationError};
use std::fmt;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

mod pool;
pub use pool::{ManagedHostEffectStagePool, ManagedHostEffectStageSlot};

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

struct LedgerState {
    snapshot: ManagedHostEffectLedgerSnapshot,
}

#[derive(Clone)]
pub struct ManagedHostEffectLedger {
    state: Arc<Mutex<LedgerState>>,
}

impl ManagedHostEffectLedger {
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
                unallocated: Mutex::new(bytes),
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
            Ok(HostEffectLiveAllocationClaim::Owned(Box::from_raw(pointer)))
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

impl HostEffectLiveAllocationReservation for TemporaryClaim {}

impl Drop for TemporaryClaim {
    fn drop(&mut self) {
        let mut temporary = self.stage.temporary.lock().expect("temporary ledger lock");
        let mut ledger = self.stage.ledger.lock().expect("host ledger lock");
        temporary.live_bytes -= self.bytes;
        ledger.snapshot.live_claimed_bytes -= self.bytes;
        if temporary.live_bytes == 0 && temporary.retire_pending {
            let reserved = self.stage.reserved_bytes.swap(0, Ordering::AcqRel);
            ledger.snapshot.live_reserved_bytes -= reserved;
            temporary.retire_pending = false;
        }
    }
}

struct OutputClaim {
    ledger: Arc<Mutex<LedgerState>>,
    headroom: std::sync::Weak<HeadroomInner>,
    bytes: u64,
}

struct ManagedOutputHeadroom {
    inner: Arc<HeadroomInner>,
}

struct HeadroomInner {
    ledger: Arc<Mutex<LedgerState>>,
    unallocated: Mutex<u64>,
}

impl fmt::Debug for ManagedOutputHeadroom {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedOutputHeadroom")
            .finish_non_exhaustive()
    }
}

impl HostEffectOutputHeadroom for ManagedOutputHeadroom {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        let bytes =
            u64::try_from(bytes).map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        let mut remaining = self.inner.unallocated.lock().expect("output headroom lock");
        if bytes > *remaining {
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
        *remaining -= bytes;
        ledger.snapshot.live_claimed_bytes = claimed;
        ledger.snapshot.peak_claimed_bytes = ledger.snapshot.peak_claimed_bytes.max(claimed);
        Ok(Arc::new(OutputClaim {
            ledger: self.inner.ledger.clone(),
            headroom: Arc::downgrade(&self.inner),
            bytes,
        }))
    }
}

impl Drop for HeadroomInner {
    fn drop(&mut self) {
        let remaining = *self.unallocated.lock().expect("output headroom lock");
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
            let mut remaining = headroom.unallocated.lock().expect("output headroom lock");
            let mut state = self.ledger.lock().expect("host ledger lock");
            *remaining += self.bytes;
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
