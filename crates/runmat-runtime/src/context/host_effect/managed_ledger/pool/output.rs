//! Setup-owned exact output headroom for one reusable staged slot. The slot
//! cannot be reacquired while a value retains one of its allocation claims.

use super::*;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostAllocationReservation,
    ExactHostNumericAllocationError,
};
use std::alloc::{alloc, Layout};
use std::sync::{Arc, Weak};

use super::super::{OwnedOutputClaim, OwnedOutputClaimRefund};

struct OutputState {
    remaining: u64,
    active: bool,
    cancellation: Option<runmat_execution::cancellation::CancellationToken>,
    session_owner: Option<Arc<AtomicU64>>,
    session_generation: u64,
}

pub(super) struct PooledOutputPermit {
    owner: Weak<Self>,
    ledger: Arc<Mutex<LedgerState>>,
    active_epoch: Arc<AtomicBool>,
    state: Mutex<OutputState>,
}

impl PooledOutputPermit {
    pub(super) fn transfer_settled_stage_bytes(
        &self,
        stage: &ManagedHostEffectStageReservation,
        bytes: u64,
        epoch: u64,
    ) -> Result<(), HostEffectAllocationError> {
        if !Arc::ptr_eq(&stage.inner.ledger, &self.ledger) {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        let temporary = stage.inner.temporary.lock().expect("temporary ledger lock");
        let mut output = self.state.lock().expect("pooled output lock");
        let ledger = stage.inner.ledger.lock().expect("host ledger lock");
        if !ledger.snapshot.current || ledger.snapshot.epoch != epoch {
            return Err(HostEffectAllocationError::Revoked);
        }
        if output
            .cancellation
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if !output.active
            || !self.active_epoch.load(Ordering::Acquire)
            || temporary.retire_pending
            || output.cancellation.is_none()
            || output
                .session_owner
                .as_ref()
                .is_none_or(|owner| owner.load(Ordering::Acquire) != output.session_generation)
        {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        let reserved = stage.inner.reserved_bytes.load(Ordering::Acquire);
        let remaining = reserved
            .checked_sub(bytes)
            .ok_or(HostEffectAllocationError::ExceedsStageEnvelope)?;
        if remaining < temporary.live_bytes {
            return Err(HostEffectAllocationError::ExceedsStageEnvelope);
        }
        let next_output = output
            .remaining
            .checked_add(bytes)
            .ok_or(HostEffectAllocationError::Overflow)?;
        stage
            .inner
            .reserved_bytes
            .store(remaining, Ordering::Release);
        output.remaining = next_output;
        Ok(())
    }

    pub(super) fn new(
        ledger: &ManagedHostEffectLedger,
        active_epoch: Arc<AtomicBool>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|owner| Self {
            owner: owner.clone(),
            ledger: ledger.state.clone(),
            active_epoch,
            state: Mutex::new(OutputState {
                remaining: 0,
                active: false,
                cancellation: None,
                session_owner: None,
                session_generation: 0,
            }),
        })
    }

    pub(super) fn prepare(
        &self,
        bytes: u64,
        cancellation: runmat_execution::cancellation::CancellationToken,
        session_owner: Arc<AtomicU64>,
        session_generation: u64,
    ) -> Result<(), HostEffectAllocationError> {
        let mut output = self.state.lock().expect("pooled output lock");
        if output.active || output.remaining != 0 {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        if !self.active_epoch.load(Ordering::Acquire) || cancellation.is_cancelled() {
            return Err(HostEffectAllocationError::Cancelled);
        }
        if session_owner.load(Ordering::Acquire) != session_generation {
            return Err(HostEffectAllocationError::Revoked);
        }
        let mut ledger = self.ledger.lock().expect("host ledger lock");
        if !ledger.snapshot.current {
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
        ledger.snapshot.live_reserved_bytes = next;
        ledger.snapshot.peak_reserved_bytes = ledger.snapshot.peak_reserved_bytes.max(next);
        output.remaining = bytes;
        output.active = true;
        output.cancellation = Some(cancellation);
        output.session_owner = Some(session_owner);
        output.session_generation = session_generation;
        Ok(())
    }

    pub(super) fn retire(&self) {
        let mut output = self.state.lock().expect("pooled output lock");
        if !output.active {
            return;
        }
        let mut ledger = self.ledger.lock().expect("host ledger lock");
        ledger.snapshot.live_reserved_bytes -= output.remaining;
        output.remaining = 0;
        output.active = false;
        output.cancellation = None;
        output.session_owner = None;
    }

    fn claim(&self, bytes: usize) -> Result<(Arc<Self>, u64), ExactHostNumericAllocationError> {
        let charged = bytes
            .checked_add(OwnedOutputClaim::metadata_bytes())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(ExactHostNumericAllocationError::ExceedsPermit)?;
        let mut output = self.state.lock().expect("pooled output lock");
        if !output.active
            || !self.active_epoch.load(Ordering::Acquire)
            || output
                .cancellation
                .as_ref()
                .is_none_or(|token| token.is_cancelled())
            || output
                .session_owner
                .as_ref()
                .is_none_or(|owner| owner.load(Ordering::Acquire) != output.session_generation)
            || charged > output.remaining
        {
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        let mut ledger = self.ledger.lock().expect("host ledger lock");
        if !ledger.snapshot.current {
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        let claimed = ledger
            .snapshot
            .live_claimed_bytes
            .checked_add(charged)
            .ok_or(ExactHostNumericAllocationError::ExceedsPermit)?;
        let owner = self
            .owner
            .upgrade()
            .ok_or(ExactHostNumericAllocationError::ExceedsPermit)?;
        output.remaining -= charged;
        ledger.snapshot.live_claimed_bytes = claimed;
        ledger.snapshot.peak_claimed_bytes = ledger.snapshot.peak_claimed_bytes.max(claimed);
        Ok((owner, charged))
    }

    /// # Safety
    /// A non-null allocator result must be a fresh, uniquely owned global
    /// allocation with exactly the supplied layout and alignment.
    unsafe fn reserve_exact_claim_with(
        &self,
        bytes: usize,
        allocate: impl FnOnce(Layout) -> *mut u8,
    ) -> Result<ExactHostAllocationClaim, ExactHostNumericAllocationError> {
        let (owner, charged) = self.claim(bytes)?;
        let owner: Arc<dyn OwnedOutputClaimRefund> = owner;
        // SAFETY: the caller supplies the exact allocator contract above.
        let boxed = unsafe { OwnedOutputClaim::try_box_with(owner, charged, allocate)? };
        let claim = ExactHostAllocationClaim::owned(boxed);
        let output = self.state.lock().expect("pooled output lock");
        let current = output.active
            && self.active_epoch.load(Ordering::Acquire)
            && output
                .cancellation
                .as_ref()
                .is_some_and(|token| !token.is_cancelled())
            && output
                .session_owner
                .as_ref()
                .is_some_and(|owner| owner.load(Ordering::Acquire) == output.session_generation);
        drop(output);
        if !current
            || !self
                .ledger
                .lock()
                .expect("host ledger lock")
                .snapshot
                .current
        {
            drop(claim);
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        Ok(claim)
    }
}

impl std::fmt::Debug for PooledOutputPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledOutputPermit").finish_non_exhaustive()
    }
}

impl ExactHostAllocationPermit for PooledOutputPermit {
    fn reserve_exact(
        &self,
        _bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        // An Arc claim would introduce an uncharged per-call control block.
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    }

    fn reserve_exact_claim(
        &self,
        bytes: usize,
    ) -> Result<ExactHostAllocationClaim, ExactHostNumericAllocationError> {
        let allocate = |layout| {
            // SAFETY: the exact layout is released by the owned claim.
            unsafe { alloc(layout) }
        };
        // SAFETY: the global allocator returns null or a unique exact block.
        unsafe { self.reserve_exact_claim_with(bytes, allocate) }
    }
}

impl OwnedOutputClaimRefund for PooledOutputPermit {
    fn refund_exact_output_claim(&self, charged: u64) {
        let mut output = self.state.lock().expect("pooled output lock");
        let mut ledger = self.ledger.lock().expect("host ledger lock");
        ledger.snapshot.live_claimed_bytes -= charged;
        if output.active {
            output.remaining += charged;
        } else {
            ledger.snapshot.live_reserved_bytes -= charged;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::cancellation::CancellationSource;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn claimed_output_outlives_retired_stage_and_fences_slot_reuse() {
        let ledger = ManagedHostEffectLedger::new(96);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let stage = pool.acquire(8).unwrap();
        let session = Arc::new(AtomicU64::new(1));
        let cancellation = CancellationSource::new();
        let permit = stage
            .reservation()
            .reserve_preallocated_output_permit(64, cancellation.token(), session, 1)
            .unwrap()
            .unwrap();
        assert_eq!(ledger.snapshot().live_reserved_bytes, 72);
        assert!(matches!(
            permit.reserve_exact(1),
            Err(ExactHostNumericAllocationError::ExceedsPermit)
        ));
        let claim = permit.reserve_exact_claim(4).expect("owned claim");
        let charged = 4 + OwnedOutputClaim::metadata_bytes() as u64;
        assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
        stage.reservation().retire_invocation();
        drop(stage);
        assert_eq!(ledger.snapshot().live_reserved_bytes, charged);
        assert!(matches!(
            pool.acquire(8),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        assert!(matches!(
            permit.reserve_exact_claim(1),
            Err(ExactHostNumericAllocationError::ExceedsPermit)
        ));
        drop(permit);
        assert!(matches!(
            pool.acquire(8),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        drop(claim);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        let reused = pool.acquire(8).expect("final output owner released");
        assert_eq!(reused.generation, 2);
        drop(reused);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cancellation_and_one_session_rebind_do_not_poison_peer_output() {
        let ledger = ManagedHostEffectLedger::new(160);
        let first_pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let second_pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let first = first_pool.acquire(8).unwrap();
        let second = second_pool.acquire(8).unwrap();
        let first_session = Arc::new(AtomicU64::new(1));
        let second_session = Arc::new(AtomicU64::new(1));
        let cancelled = CancellationSource::new();
        let first_permit = first
            .reservation()
            .reserve_preallocated_output_permit(64, cancelled.token(), first_session.clone(), 1)
            .unwrap()
            .unwrap();
        let second_permit = second
            .reservation()
            .reserve_preallocated_output_permit(
                64,
                CancellationSource::new().token(),
                second_session,
                1,
            )
            .unwrap()
            .unwrap();
        first_session.store(2, Ordering::Release);
        cancelled.cancel(runmat_execution::CancellationReason::User);
        assert!(matches!(
            first_permit.reserve_exact_claim(1),
            Err(ExactHostNumericAllocationError::ExceedsPermit)
        ));
        let peer = second_permit
            .reserve_exact_claim(1)
            .expect("peer remains current");
        drop(first);
        drop(first_permit);
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        drop(second);
        drop(second_permit);
        drop(peer);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn denied_claim_metadata_allocation_refunds_and_allows_reuse() {
        let ledger = ManagedHostEffectLedger::new(96);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let stage = pool.acquire(8).unwrap();
        let session = Arc::new(AtomicU64::new(1));
        stage
            .reservation()
            .reserve_preallocated_output_permit(64, CancellationSource::new().token(), session, 1)
            .unwrap()
            .unwrap();
        let output = &pool.slots[0].output;
        let before = ledger.snapshot();
        // SAFETY: null never transfers a physical allocation into the claim.
        let denied = unsafe { output.reserve_exact_claim_with(4, |_| std::ptr::null_mut()) };
        assert!(matches!(
            denied,
            Err(ExactHostNumericAllocationError::AllocationFailed)
        ));
        assert_eq!(
            ledger.snapshot().live_reserved_bytes,
            before.live_reserved_bytes
        );
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        let claim = output.reserve_exact_claim(4).expect("refunded headroom");
        drop(claim);
        drop(stage);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn owned_claim_header_retires_before_its_funded_bytes_are_refunded() {
        let ledger = ManagedHostEffectLedger::new(96);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let stage = pool.acquire(8).unwrap();
        let session = Arc::new(AtomicU64::new(1));
        stage
            .reservation()
            .reserve_preallocated_output_permit(64, CancellationSource::new().token(), session, 1)
            .unwrap()
            .unwrap();
        let output = &pool.slots[0].output;
        let (owner, charged) = output.claim(4).expect("owned claim");
        let owner: Arc<dyn OwnedOutputClaimRefund> = owner;
        // SAFETY: the global allocator returns the exact requested block.
        let claim =
            unsafe { OwnedOutputClaim::try_box_with(owner, charged, |layout| alloc(layout)) }
                .expect("owned claim header");
        // SAFETY: the test Box owns one exact OwnedOutputClaim allocation;
        // the callback releases it once with its original layout.
        unsafe {
            OwnedOutputClaim::retire_owned_with(claim, |pointer, layout| {
                assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
                std::alloc::dealloc(pointer.as_ptr(), layout);
            })
        };
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        drop(stage);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn retirement_while_claim_metadata_allocation_is_pending_refunds_once() {
        use std::sync::Barrier;

        let ledger = ManagedHostEffectLedger::new(96);
        let pool = ManagedHostEffectStagePool::new(&ledger, 1).unwrap();
        let stage = pool.acquire(8).unwrap();
        let session = Arc::new(AtomicU64::new(1));
        stage
            .reservation()
            .reserve_preallocated_output_permit(64, CancellationSource::new().token(), session, 1)
            .unwrap()
            .unwrap();
        let output = Arc::clone(&pool.slots[0].output);
        let entered = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        std::thread::scope(|scope| {
            let entered_worker = Arc::clone(&entered);
            let resume_worker = Arc::clone(&resume);
            let worker = scope.spawn(move || {
                // SAFETY: the closure returns a fresh global-allocator block
                // with the exact layout after the deterministic pause.
                let allocate = |layout| {
                    entered_worker.wait();
                    resume_worker.wait();
                    // SAFETY: this is a fresh exact-layout global allocation.
                    unsafe { alloc(layout) }
                };
                unsafe { output.reserve_exact_claim_with(4, allocate) }
            });
            entered.wait();
            assert!(ledger.snapshot().live_claimed_bytes > 0);
            stage.reservation().retire_invocation();
            drop(stage);
            resume.wait();
            assert!(matches!(
                worker.join().expect("claim worker"),
                Err(ExactHostNumericAllocationError::ExceedsPermit)
            ));
        });
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        let next = pool.acquire(8).expect("retired slot and metadata refunded");
        assert_eq!(next.generation, 2);
    }
}
