use super::{
    catch_allocation_cleanup_unwind, AllocationIdentity, AllocationOwner,
    AllocationRegistrationError, AllocationReleaseError, AllocationReleaseOutcome,
    AllocationRetirementOutcome, GpuPhysicalDescriptor, GpuViewError,
};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

struct LeaseInner {
    identity: AllocationIdentity,
    owner: Arc<dyn AllocationOwner>,
    armed: AtomicBool,
    physical: OnceLock<GpuPhysicalDescriptor>,
    next_view: AtomicU64,
}

impl LeaseInner {
    fn retire(&self) -> Option<AllocationRetirementOutcome> {
        if !self.armed.swap(false, Ordering::AcqRel) {
            return None;
        }
        // The owner persists the typed outcome. The primitive never retries a
        // physical release, including when its result is uncertain.
        match catch_allocation_cleanup_unwind(|| self.owner.retire(self.identity)) {
            Ok(outcome) => Some(outcome),
            Err(_) => {
                // Also contain a broken recovery implementation during an existing
                // unwind. Such an owner violates the documented persistence contract;
                // this is not a successful retirement or a reason to retry.
                let _ = catch_allocation_cleanup_unwind(|| {
                    self.owner.record_retirement_panic(self.identity);
                });
                Some(AllocationRetirementOutcome::Uncertain {
                    detail: "allocation retirement unwound".into(),
                })
            }
        }
    }
}
impl Drop for LeaseInner {
    fn drop(&mut self) {
        let _ = self.retire();
    }
}

/// One owning reference to a registered allocation. Clone copies only an Arc.
/// Shape/storage views belong to the eventual tensor carrier, not this lease.
/// Retirement callback unwinds are contained on unwind-enabled targets. This
/// does not make an arbitrary owner's own destructor nonpanicking: the owner
/// must satisfy its documented teardown contract. Panic-abort cannot be caught.
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
                physical: OnceLock::new(),
                next_view: AtomicU64::new(1),
            }),
        };
        let physical = lease
            .inner
            .owner
            .register_lease(identity, lease.downgrade())?;
        // No acquiring thread may observe an armed lease without its immutable
        // payload facts. Only this registration path initializes the slot.
        lease.inner.physical.get_or_init(|| physical);
        lease.inner.armed.store(true, Ordering::Release);
        Ok(lease)
    }

    pub fn identity(&self) -> AllocationIdentity {
        self.inner.identity
    }

    pub fn physical(&self) -> &GpuPhysicalDescriptor {
        self.inner
            .physical
            .get()
            .expect("published allocation lease has physical facts")
    }

    pub fn validate_live(&self) -> Result<(), super::AllocationAcquisitionError> {
        self.inner.owner.validate_live(self.identity())
    }

    pub(super) fn validate_view(
        &self,
        view: &super::GpuTensorView,
    ) -> Result<(), super::AllocationAcquisitionError> {
        self.inner.owner.validate_view(self.identity(), view)
    }

    pub(super) fn next_view_identity(&self) -> Result<u64, GpuViewError> {
        self.inner
            .next_view
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| GpuViewError::IdentityExhausted)
    }

    /// Atomic ownership extraction, never a racy strong_count/free sequence.
    /// Losing to a concurrent weak upgrade releases only this reference.
    pub fn release(self) -> Result<AllocationReleaseOutcome, AllocationReleaseError> {
        match Arc::try_unwrap(self.inner) {
            Err(reference) => {
                drop(reference);
                Ok(AllocationReleaseOutcome::ReferenceReleased)
            }
            Ok(inner) => match inner.retire() {
                Some(AllocationRetirementOutcome::Retired) => Ok(AllocationReleaseOutcome::Retired),
                Some(outcome) => Err(AllocationReleaseError::new(
                    inner.identity,
                    outcome,
                    inner.owner.clone(),
                )),
                None => Ok(AllocationReleaseOutcome::ReferenceReleased),
            },
        }
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
    /// Upgrade an already registered allocation and validate non-owning wire
    /// facts. The provider locates this weak record through its existing map.
    /// Clone the weak association and release that map's lock before calling:
    /// live validation re-enters the captured owner, never an ambient provider.
    pub fn acquire_tensor(
        &self,
        wire: &super::GpuTensorWireDescriptor,
    ) -> Result<super::GpuTensorHandle, super::AllocationAcquisitionError> {
        use super::AllocationAcquisitionError as Error;
        let lease = self.acquire().ok_or(Error::NotLive)?;
        if lease.identity().device_id() != wire.allocation.device_id() {
            return Err(Error::WrongOwner);
        }
        if lease.identity() != wire.allocation {
            return Err(Error::StaleGeneration);
        }
        if lease.physical() != &wire.physical {
            return Err(Error::PhysicalMismatch);
        }
        super::GpuTensorHandle::from_allocation(lease, wire.view.clone())
    }
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
