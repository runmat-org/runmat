//! Concurrent registration capacity retained by physical provider records.
//! The provider's registry lock serializes acquisition, removal and return.
use super::{ExactShared, ProviderHostMetadataError, RetainedHostAllocationClaim};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct State {
    owner: usize,
    available: AtomicUsize,
    outstanding: Arc<AtomicUsize>,
}

impl Drop for State {
    fn drop(&mut self) {
        // Live records retain this owner; only unused capacity remains here.
        self.outstanding
            .fetch_sub(self.available.load(Ordering::Acquire), Ordering::AcqRel);
    }
}

/// Capacity for serial operations which repeatedly produce and retire records.
/// This is not permission to exceed the concurrent limit.
#[derive(Clone)]
pub struct ProviderRegistrationSlotPool {
    state: ExactShared<State>,
    // Funding drops after deallocation of the final shared control block.
    _funding: Arc<dyn RetainedHostAllocationClaim>,
}

impl std::fmt::Debug for ProviderRegistrationSlotPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderRegistrationSlotPool")
            .field("owner", &self.state.get().owner)
            .field(
                "available",
                &self.state.get().available.load(Ordering::Acquire),
            )
            .finish_non_exhaustive()
    }
}

impl ProviderRegistrationSlotPool {
    pub fn requested_host_bytes() -> Option<u64> {
        u64::try_from(ExactShared::<State>::requested_bytes()).ok()
    }

    /// The provider has already added `capacity` to `outstanding` under its
    /// registry lock. Funding covers the quoted control allocation.
    pub fn from_reserved_capacity(
        owner: usize,
        capacity: usize,
        outstanding: Arc<AtomicUsize>,
        funding: Arc<dyn RetainedHostAllocationClaim>,
    ) -> Result<Self, ProviderHostMetadataError> {
        if capacity == 0 || outstanding.load(Ordering::Acquire) < capacity {
            return Err(ProviderHostMetadataError::OwnerFailure);
        }
        Ok(Self {
            state: ExactShared::new(State {
                owner,
                available: AtomicUsize::new(capacity),
                outstanding,
            }),
            _funding: funding,
        })
    }

    /// Call under the captured provider registry lock immediately before
    /// inserting the physical record. Acquisition itself does not allocate.
    pub fn acquire(
        &self,
        owner: usize,
    ) -> Result<ProviderRegistrationSlot, ProviderHostMetadataError> {
        let state = self.state.get();
        if owner != state.owner {
            return Err(ProviderHostMetadataError::OwnerFailure);
        }
        state
            .available
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .map_err(|_| ProviderHostMetadataError::OwnerFailure)?;
        if state
            .outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .is_err()
        {
            state.available.fetch_add(1, Ordering::AcqRel);
            return Err(ProviderHostMetadataError::OwnerFailure);
        }
        Ok(ProviderRegistrationSlot {
            pool: Some(self.clone()),
        })
    }
}

/// Move-only capacity held by a physical record, not by a handle view.
#[derive(Debug)]
pub struct ProviderRegistrationSlot {
    pool: Option<ProviderRegistrationSlotPool>,
}

impl ProviderRegistrationSlot {
    /// Return under the registry lock when removing the physical record.
    /// Taking the pool also drops its control owner before an empty-table
    /// check, so another reservation cannot steal temporarily uncounted space.
    pub fn release(&mut self) {
        if let Some(pool) = self.pool.take() {
            pool.state.get().outstanding.fetch_add(1, Ordering::AcqRel);
            pool.state.get().available.fetch_add(1, Ordering::AcqRel);
        }
    }
}

impl Drop for ProviderRegistrationSlot {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_recycles_after_retirement_and_outlives_admission() {
        let outstanding = Arc::new(AtomicUsize::new(2));
        let pool = ProviderRegistrationSlotPool::from_reserved_capacity(
            7,
            2,
            outstanding.clone(),
            Arc::new(()),
        )
        .unwrap();
        assert!(pool.acquire(8).is_err());
        let mut first = pool.acquire(7).unwrap();
        let second = pool.acquire(7).unwrap();
        assert!(pool.acquire(7).is_err());
        assert_eq!(outstanding.load(Ordering::Acquire), 0);
        first.release();
        first.release();
        let last = pool.acquire(7).unwrap();
        assert_eq!(outstanding.load(Ordering::Acquire), 0);
        drop(pool);
        drop(second);
        assert_eq!(outstanding.load(Ordering::Acquire), 1);
        drop(last);
        assert_eq!(outstanding.load(Ordering::Acquire), 0);
    }
}
