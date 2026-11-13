//! Session-owned, fixed-capacity in-flight observation identities.
//!
//! This is a requested-allocation backing owner, not a resource certificate
//! for a complete builtin. Ordinary external placement retains its dynamic
//! observation registry until a route has a matching managed admission.

use super::host_effect::{
    HostEffectAllocationError, ManagedHostEffectLedger, ManagedHostEffectStageReservation,
};
use runmat_execution::{Digest, PlacementInvalidation, PlacementPermitIdentity};
use std::sync::{Arc, Mutex};

#[derive(Default, Clone, Copy)]
struct Slot {
    generation: u64,
    identity: Option<PlacementPermitIdentity>,
    candidate: Option<Digest>,
    current: bool,
}

struct PoolState {
    slots: Vec<Slot>,
}

struct PoolOwner {
    state: Mutex<PoolState>,
    ledger: ManagedHostEffectLedger,
    /// This reservation is retained for exactly the lifetime of the backing.
    _backing: Arc<ManagedHostEffectStageReservation>,
}

/// Preallocated observation storage. The backing reservation covers both the
/// requested slot array and at most one callback-shared lease owner per slot.
/// No second lease can be issued for a live slot.
pub struct ManagedObservationPool {
    owner: Arc<PoolOwner>,
}

pub struct ManagedObservationLease {
    owner: Arc<PoolOwner>,
    slot: usize,
    generation: u64,
}

impl ManagedObservationPool {
    /// Requested slot backing plus one Arc lease allocation for every
    /// concurrently usable slot. The lease can outlive its caller while
    /// callback-owned provider work is still physically retiring.
    pub fn requested_backing_bytes(capacity: usize) -> Result<u64, HostEffectAllocationError> {
        let lease_bytes = std::mem::size_of::<ManagedObservationLease>()
            .checked_add(std::mem::size_of::<usize>() * 2)
            .and_then(|bytes| bytes.checked_add(std::mem::align_of::<ManagedObservationLease>()))
            .ok_or(HostEffectAllocationError::Overflow)?;
        capacity
            .checked_mul(
                std::mem::size_of::<Slot>()
                    .checked_add(lease_bytes)
                    .ok_or(HostEffectAllocationError::Overflow)?,
            )
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(HostEffectAllocationError::Overflow)
    }

    #[cfg(test)]
    pub(crate) fn synthetic_unaccounted_for_tests(
        ledger: &ManagedHostEffectLedger,
    ) -> Result<Self, HostEffectAllocationError> {
        // Test placement fakes exercise receipt ownership against tiny ledger
        // capacities; this is never a production reservation or certificate.
        let backing = ledger.reserve_stage(0)?;
        Ok(Self {
            owner: Arc::new(PoolOwner {
                state: Mutex::new(PoolState {
                    slots: vec![Slot::default()],
                }),
                ledger: ledger.clone(),
                _backing: backing,
            }),
        })
    }

    pub fn new(
        ledger: &ManagedHostEffectLedger,
        capacity: usize,
    ) -> Result<Self, HostEffectAllocationError> {
        let bytes = Self::requested_backing_bytes(capacity)?;
        let backing = ledger.reserve_stage(bytes)?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(capacity)
            .map_err(|_| HostEffectAllocationError::AllocationFailed)?;
        slots.resize_with(capacity, Slot::default);
        Ok(Self {
            owner: Arc::new(PoolOwner {
                state: Mutex::new(PoolState { slots }),
                ledger: ledger.clone(),
                _backing: backing,
            }),
        })
    }

    pub fn acquire(
        &self,
        identity: PlacementPermitIdentity,
        candidate: Digest,
    ) -> Result<Arc<ManagedObservationLease>, HostEffectAllocationError> {
        if !self.owner.ledger.snapshot().current {
            return Err(HostEffectAllocationError::Revoked);
        }
        let mut state = self.owner.state.lock().expect("managed observation lock");
        let (slot, entry) = state
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, entry)| entry.identity.is_none())
            .ok_or(HostEffectAllocationError::ExceedsCapacity)?;
        entry.generation = entry
            .generation
            .checked_add(1)
            .ok_or(HostEffectAllocationError::Overflow)?;
        entry.identity = Some(identity);
        entry.candidate = Some(candidate);
        entry.current = true;
        Ok(Arc::new(ManagedObservationLease {
            owner: Arc::clone(&self.owner),
            slot,
            generation: entry.generation,
        }))
    }

    pub fn has_active(&self, identity: PlacementPermitIdentity, candidate: Digest) -> bool {
        let state = self.owner.state.lock().expect("managed observation lock");
        state
            .slots
            .iter()
            .any(|entry| entry.identity == Some(identity) && entry.candidate == Some(candidate))
    }

    pub fn invalidate(&self, invalidation: &PlacementInvalidation) {
        let mut state = self.owner.state.lock().expect("managed observation lock");
        for entry in &mut state.slots {
            if entry
                .identity
                .as_ref()
                .is_some_and(|identity| invalidation.matches_permit_identity(identity))
            {
                entry.current = false;
            }
        }
    }
}

impl ManagedObservationLease {
    pub fn is_current(&self) -> bool {
        if !self.owner.ledger.snapshot().current {
            return false;
        }
        let state = self.owner.state.lock().expect("managed observation lock");
        let entry = &state.slots[self.slot];
        entry.generation == self.generation && entry.identity.is_some() && entry.current
    }

    pub fn same_owner(&self, pool: &ManagedObservationPool) -> bool {
        Arc::ptr_eq(&self.owner, &pool.owner)
    }

    pub(crate) fn same_ledger(&self, ledger: &ManagedHostEffectLedger) -> bool {
        self.owner.ledger.same_owner(ledger)
    }
}

impl Drop for ManagedObservationLease {
    fn drop(&mut self) {
        let mut state = self.owner.state.lock().expect("managed observation lock");
        let entry = &mut state.slots[self.slot];
        if entry.generation == self.generation {
            entry.identity = None;
            entry.candidate = None;
            entry.current = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{Digest, PlacementExecutionAuthorityRevision};

    fn identity() -> PlacementPermitIdentity {
        let digest = Digest::sha256(b"managed observation");
        PlacementPermitIdentity {
            signature: digest,
            program: None,
            authority: PlacementExecutionAuthorityRevision::HostService(digest),
            provider_provenance: None,
            policy: digest,
        }
    }

    #[test]
    fn setup_charges_requested_backing_and_releases_on_last_lease() {
        let bytes = ManagedObservationPool::requested_backing_bytes(1).unwrap();
        let ledger = ManagedHostEffectLedger::new(bytes);
        let pool = ManagedObservationPool::new(&ledger, 1).expect("one physical slot");
        assert_eq!(ledger.snapshot().live_reserved_bytes, bytes);
        let lease = pool
            .acquire(identity(), identity().signature)
            .expect("first lease");
        assert!(lease.same_owner(&pool));
        drop(pool);
        assert_eq!(ledger.snapshot().live_reserved_bytes, bytes);
        let callback_held = Arc::clone(&lease);
        drop(lease);
        assert!(callback_held.is_current());
        assert_eq!(ledger.snapshot().live_reserved_bytes, bytes);
        drop(callback_held);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn exhausted_and_foreign_slots_cannot_be_reused_until_drop() {
        let bytes = ManagedObservationPool::requested_backing_bytes(2).unwrap();
        let first = ManagedObservationPool::new(&ManagedHostEffectLedger::new(bytes), 1).unwrap();
        let other = ManagedObservationPool::new(&ManagedHostEffectLedger::new(bytes), 1).unwrap();
        let lease = first.acquire(identity(), identity().signature).unwrap();
        assert!(!lease.same_owner(&other));
        assert!(matches!(
            first.acquire(identity(), identity().signature),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        let callback_held = Arc::clone(&lease);
        drop(lease);
        assert!(matches!(
            first.acquire(identity(), identity().signature),
            Err(HostEffectAllocationError::ExceedsCapacity)
        ));
        drop(callback_held);
        assert!(first
            .acquire(identity(), identity().signature)
            .unwrap()
            .is_current());
    }

    #[test]
    fn matching_invalidation_fences_late_observation_and_reuse_has_new_generation() {
        let bytes = ManagedObservationPool::requested_backing_bytes(1).unwrap();
        let pool = ManagedObservationPool::new(&ManagedHostEffectLedger::new(bytes), 1).unwrap();
        let first = pool.acquire(identity(), identity().signature).unwrap();
        let first_generation = first.generation;
        pool.invalidate(&PlacementInvalidation::All);
        assert!(!first.is_current());
        drop(first);
        let next = pool.acquire(identity(), identity().signature).unwrap();
        assert!(next.is_current());
        assert_ne!(next.generation, first_generation);
    }

    #[test]
    fn provider_provenance_invalidates_matching_managed_lease_only() {
        let bytes = ManagedObservationPool::requested_backing_bytes(2).unwrap();
        let pool = ManagedObservationPool::new(&ManagedHostEffectLedger::new(bytes), 2).unwrap();
        let provider_a = Digest::sha256(b"provider A");
        let provider_b = Digest::sha256(b"provider B");
        let mut first_identity = identity();
        first_identity.authority =
            PlacementExecutionAuthorityRevision::RuntimeBindingSession(Digest::sha256(b"VM"));
        first_identity.provider_provenance = Some(provider_a);
        let mut second_identity = first_identity;
        second_identity.provider_provenance = Some(provider_b);
        let first = pool
            .acquire(first_identity, first_identity.signature)
            .unwrap();
        let second = pool
            .acquire(second_identity, second_identity.signature)
            .unwrap();
        pool.invalidate(&PlacementInvalidation::Provider { digest: provider_a });
        assert!(!first.is_current());
        assert!(second.is_current());
    }

    #[test]
    fn ledger_revocation_fences_existing_and_new_slots() {
        let bytes = ManagedObservationPool::requested_backing_bytes(1).unwrap();
        let ledger = ManagedHostEffectLedger::new(bytes);
        let pool = ManagedObservationPool::new(&ledger, 1).unwrap();
        let lease = pool.acquire(identity(), identity().signature).unwrap();
        ledger.revoke();
        assert!(!lease.is_current());
        drop(lease);
        assert!(matches!(
            pool.acquire(identity(), identity().signature),
            Err(HostEffectAllocationError::Revoked)
        ));
    }
}
