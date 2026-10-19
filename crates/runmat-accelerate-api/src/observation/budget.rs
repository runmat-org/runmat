//! Provider-owned bound on in-flight instrumentation, not device scheduling.
use std::num::NonZeroU32;
#[cfg(feature = "wgpu")]
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub struct ProviderObservationBudget {
    limit: NonZeroU32,
    used: AtomicU32,
    #[cfg(feature = "wgpu")]
    query_requested_byte_limit: u64,
    #[cfg(feature = "wgpu")]
    query_requested_bytes: AtomicU64,
    #[cfg(feature = "wgpu")]
    query_poisoned: AtomicBool,
}
impl ProviderObservationBudget {
    pub fn new(limit: NonZeroU32) -> Self {
        Self {
            limit,
            used: AtomicU32::new(0),
            #[cfg(feature = "wgpu")]
            query_requested_byte_limit: 0,
            #[cfg(feature = "wgpu")]
            query_requested_bytes: AtomicU64::new(0),
            #[cfg(feature = "wgpu")]
            query_poisoned: AtomicBool::new(false),
        }
    }
    pub fn limit(&self) -> u32 {
        self.limit.get()
    }
    pub(super) fn acquire(self: &Arc<Self>) -> Option<Permit> {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                (used < self.limit.get()).then(|| used + 1)
            })
            .ok()?;
        Some(Permit(self.clone()))
    }

    /// Opt in only at a provider with a checked bound for optional query,
    /// resolve, and staging demand. Required submission slots remain separate:
    /// an optional query must never make an admitted receipt fail later.
    #[cfg(feature = "wgpu")]
    pub fn with_wgpu_query_bytes(limit: NonZeroU32, query_requested_byte_limit: u64) -> Self {
        Self {
            limit,
            used: AtomicU32::new(0),
            query_requested_byte_limit,
            query_requested_bytes: AtomicU64::new(0),
            query_poisoned: AtomicBool::new(false),
        }
    }

    #[cfg(feature = "wgpu")]
    pub fn wgpu_query_requested_byte_limit(&self) -> u64 {
        self.query_requested_byte_limit
    }

    /// Reserve prospective requested bytes *before* optional device allocation.
    /// Failure disables device timing only; it cannot consume the mandatory
    /// provider submission capacity or alter semantic execution.
    #[cfg(feature = "wgpu")]
    pub fn try_reserve_wgpu_query(
        self: &Arc<Self>,
        requested_bytes: u64,
    ) -> Option<WgpuQueryPermit> {
        if requested_bytes == 0 || self.query_poisoned.load(Ordering::SeqCst) {
            return None;
        }
        self.query_requested_bytes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(requested_bytes)
                    .filter(|next| *next <= self.query_requested_byte_limit)
            })
            .ok()?;
        let permit = WgpuQueryPermit {
            budget: self.clone(),
            requested_bytes,
        };
        // The post-reservation check orders a concurrent poison against
        // admission. A permit whose check precedes poison was admitted while
        // the budget was still live and remains bounded by the byte quota;
        // no reservation can linearize after the poison transition.
        if self.query_poisoned.load(Ordering::SeqCst) {
            return None;
        }
        Some(permit)
    }

    /// A browser queue-tail map consumes one in-flight observation slot and
    /// one exact fence allocation. Both permits drop together on completion,
    /// cancellation, or abandoned mapping.
    #[cfg(feature = "wgpu")]
    pub fn try_reserve_wgpu_tail(self: &Arc<Self>, requested_bytes: u64) -> Option<WgpuTailPermit> {
        let slot = self.acquire()?;
        let bytes = self.try_reserve_wgpu_query(requested_bytes)?;
        Some(WgpuTailPermit {
            _slot: slot,
            _bytes: bytes,
        })
    }
}
pub(super) struct Permit(Arc<ProviderObservationBudget>);

/// An in-flight device query/readback reservation. Its Drop releases capacity
/// on success, cancellation, map failure, and abandoned emission alike.
#[cfg(feature = "wgpu")]
pub struct WgpuQueryPermit {
    budget: Arc<ProviderObservationBudget>,
    requested_bytes: u64,
}

/// Joint callback-capacity and requested-byte owner for a browser tail.
#[cfg(feature = "wgpu")]
pub struct WgpuTailPermit {
    _slot: Permit,
    _bytes: WgpuQueryPermit,
}

#[cfg(feature = "wgpu")]
impl WgpuTailPermit {
    /// A failed map cannot certify physical retirement. Do not recycle
    /// optional tail capacity onto the same uncertain device route.
    pub fn poison_budget(&self) {
        self._bytes.poison_budget();
    }
}

#[cfg(feature = "wgpu")]
impl WgpuQueryPermit {
    /// A failed exact-submission retirement can no longer safely recycle
    /// optional query capacity. Disable future optional queries before this
    /// permit is dropped; mandatory provider submissions remain admissible.
    pub fn poison_budget(&self) {
        self.budget.query_poisoned.store(true, Ordering::SeqCst);
    }
}

#[cfg(feature = "wgpu")]
impl Drop for WgpuQueryPermit {
    fn drop(&mut self) {
        self.budget
            .query_requested_bytes
            .fetch_sub(self.requested_bytes, Ordering::AcqRel);
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.0.used.fetch_sub(1, Ordering::AcqRel);
    }
}
