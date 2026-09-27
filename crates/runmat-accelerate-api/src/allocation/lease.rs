use super::{AllocationIdentity, AllocationOwner, AllocationRegistrationError};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

struct LeaseInner {
    identity: AllocationIdentity,
    owner: Arc<dyn AllocationOwner>,
    armed: AtomicBool,
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        if !self.armed.swap(false, Ordering::AcqRel) {
            return;
        }
        // The owner persists the typed outcome. The primitive never retries a
        // physical release, including when its result is uncertain.
        if catch_unwind(AssertUnwindSafe(|| self.owner.retire(self.identity))).is_err() {
            // Also contain a broken recovery implementation during an existing
            // unwind. Such an owner violates the documented persistence contract;
            // this is not a successful retirement or a reason to retry.
            let _ = catch_unwind(AssertUnwindSafe(|| {
                self.owner.record_retirement_panic(self.identity);
            }));
        }
    }
}

/// One owning reference to a registered allocation. Clone copies only an Arc.
/// Shape/storage views belong to the eventual tensor carrier, not this lease.
#[derive(Clone)]
pub struct AllocationLease {
    inner: Arc<LeaseInner>,
}

impl AllocationLease {
    /// Register against the captured owner before making the lease acquirable.
    /// Identity values are not enough: the owner must admit an existing payload
    /// and reject duplicate registration through its own allocation map.
    pub fn register(
        identity: AllocationIdentity,
        owner: Arc<dyn AllocationOwner>,
    ) -> Result<Self, AllocationRegistrationError> {
        let lease = Self {
            inner: Arc::new(LeaseInner {
                identity,
                owner,
                armed: AtomicBool::new(false),
            }),
        };
        lease
            .inner
            .owner
            .register_lease(identity, lease.downgrade())?;
        lease.inner.armed.store(true, Ordering::Release);
        Ok(lease)
    }

    pub fn identity(&self) -> AllocationIdentity {
        self.inner.identity
    }

    pub fn downgrade(&self) -> WeakAllocationLease {
        WeakAllocationLease {
            inner: Arc::downgrade(&self.inner),
        }
    }

    /// Descriptors with equal numbers do not prove common lifetime authority.
    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl fmt::Debug for AllocationLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AllocationLease")
            .field("identity", &self.identity())
            .finish_non_exhaustive()
    }
}

/// Non-owning map association. Does not retain payload/state on its own.
/// Acquisition cannot resurrect a final-dropped lease. An owner must also
/// validate its live state and wire descriptor before exposing an acquired view.
#[derive(Clone)]
pub struct WeakAllocationLease {
    inner: Weak<LeaseInner>,
}

impl WeakAllocationLease {
    pub fn acquire(&self) -> Option<AllocationLease> {
        let inner = self.inner.upgrade()?;
        if !inner.armed.load(Ordering::Acquire) {
            return None;
        }
        Some(AllocationLease { inner })
    }
}

impl fmt::Debug for WeakAllocationLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeakAllocationLease")
            .finish_non_exhaustive()
    }
}
