//! Reusable host-owned byte accounting, not a workload resource certificate.
//! A host adapter must still certify and install its complete-call bounds.

use super::{
    HostEffectAllocationError, HostEffectLiveAllocationKind, HostEffectLiveAllocationReservation,
    HostEffectStageReservation,
};
use runmat_value::{ExactHostAllocationReservation, ExactHostNumericAllocationError};
use std::fmt;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManagedHostEffectLedgerSnapshot {
    pub capacity_bytes: u64,
    /// Conservative precharges and retained outputs, not process RSS.
    pub live_reserved_bytes: u64,
    pub peak_reserved_bytes: u64,
    /// Exact allocations actually claimed through this API.
    pub live_claimed_bytes: u64,
    pub peak_claimed_bytes: u64,
    pub current: bool,
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
        Self {
            state: Arc::new(Mutex::new(LedgerState {
                snapshot: ManagedHostEffectLedgerSnapshot {
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
                temporary: Mutex::new(TemporaryState { live_bytes: 0 }),
            }),
        }))
    }
}

struct TemporaryState {
    live_bytes: u64,
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
    pub fn reserve_output_bytes(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, HostEffectAllocationError> {
        let bytes = u64::try_from(bytes).map_err(|_| HostEffectAllocationError::Overflow)?;
        let mut state = self.inner.ledger.lock().expect("host ledger lock");
        if !state.snapshot.current {
            return Err(HostEffectAllocationError::Revoked);
        }
        let reserved = state
            .snapshot
            .live_reserved_bytes
            .checked_add(bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        let claimed = state
            .snapshot
            .live_claimed_bytes
            .checked_add(bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        if reserved > state.snapshot.capacity_bytes {
            return Err(HostEffectAllocationError::ExceedsCapacity);
        }
        state.snapshot.live_reserved_bytes = reserved;
        state.snapshot.peak_reserved_bytes = state.snapshot.peak_reserved_bytes.max(reserved);
        state.snapshot.live_claimed_bytes = claimed;
        state.snapshot.peak_claimed_bytes = state.snapshot.peak_claimed_bytes.max(claimed);
        Ok(Arc::new(OutputClaim {
            ledger: self.inner.ledger.clone(),
            bytes,
        }))
    }
}

impl HostEffectStageReservation for ManagedHostEffectStageReservation {
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
        Ok(Arc::new(TemporaryClaim {
            stage: self.inner.clone(),
            bytes,
        }))
    }

    fn reserve_output_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.reserve_output_bytes(bytes)
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)
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
    }
}

struct OutputClaim {
    ledger: Arc<Mutex<LedgerState>>,
    bytes: u64,
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
        let mut state = self.ledger.lock().expect("host ledger lock");
        state.snapshot.live_reserved_bytes -= self.bytes;
        state.snapshot.live_claimed_bytes -= self.bytes;
    }
}

#[cfg(test)]
#[path = "managed_ledger/tests.rs"]
mod tests;
