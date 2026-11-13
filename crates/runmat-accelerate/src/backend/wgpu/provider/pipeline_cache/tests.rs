use super::*;
use runmat_execution::resource::{
    ProviderHostMetadataAdmission, ProviderHostMetadataError, RetainedHostAllocationClaim,
    RetainedHostViewGrant,
};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

#[derive(Debug, Default)]
struct Ledger {
    live: AtomicU64,
    peak: AtomicU64,
    deny: AtomicBool,
}
#[derive(Debug)]
struct Claim {
    bytes: u64,
    ledger: Arc<Ledger>,
}
impl Drop for Claim {
    fn drop(&mut self) {
        self.ledger.live.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
#[derive(Debug, Default)]
struct Admission(Arc<Ledger>);
impl ProviderHostMetadataAdmission for Admission {
    fn claim_view(
        &self,
        _: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, ProviderHostMetadataError> {
        Err(ProviderHostMetadataError::OwnerFailure)
    }
    fn claim_record(
        &self,
        bytes: u64,
    ) -> Result<Arc<dyn RetainedHostAllocationClaim>, ProviderHostMetadataError> {
        if self.0.deny.load(Ordering::SeqCst) {
            return Err(ProviderHostMetadataError::ExceedsCapacity);
        }
        let live = self.0.live.fetch_add(bytes, Ordering::SeqCst) + bytes;
        self.0.peak.fetch_max(live, Ordering::SeqCst);
        Ok(Arc::new(Claim {
            bytes,
            ledger: self.0.clone(),
        }))
    }
}
impl Admission {
    fn live(&self) -> u64 {
        self.0.live.load(Ordering::SeqCst)
    }
}

fn miss<'a, T>(
    cache: &'a Mutex<PipelineCache<T>>,
    key: u64,
    admission: &Admission,
) -> super::table::CacheReservation<'a, T> {
    match PipelineCache::reserve(cache, key, Some(admission)).unwrap() {
        CacheLookup::Miss(reservation) => reservation,
        CacheLookup::Hit(_) => panic!("expected cold reservation"),
    }
}

// These are generic first-party allocation/ownership witnesses, not a claim
// that a dummy payload certifies driver-private allocation or GPU execution.
#[test]
fn refusal_and_abandoned_miss_leave_no_unfunded_backing() {
    let cache = Mutex::new(PipelineCache::<u64>::default());
    let admission = Admission::default();
    admission.0.deny.store(true, Ordering::SeqCst);
    assert!(PipelineCache::reserve(&cache, 7, Some(&admission)).is_err());
    assert_eq!(cache.lock().unwrap().len(), 0);
    assert_eq!(admission.live(), 0);
    admission.0.deny.store(false, Ordering::SeqCst);
    let reservation = miss(&cache, 7, &admission);
    assert!(admission.live() > 0);
    drop(reservation);
    assert_eq!(admission.live(), 0);
    assert!(PipelineAllocation::<u64>::prepare(Some(&Admission({
        let ledger = Arc::new(Ledger::default());
        ledger.deny.store(true, Ordering::SeqCst);
        ledger
    })))
    .is_err());
}

#[test]
fn first_winner_and_final_alias_conserve_exact_wrapper_charge() {
    // Force a real quote/admit overlap, not merely two already completed
    // reservations. A different-key winner must trigger requoting; a same-key
    // winner must still be reused. Even refusal of an obsolete quote is not a
    // refusal of the fresh, already-funded table geometry.
    struct OverlapAdmission {
        admission: Admission,
        failure: Option<ProviderHostMetadataError>,
        first: AtomicBool,
        quoted: std::sync::Barrier,
        release: std::sync::Barrier,
        cache: std::sync::Weak<Mutex<PipelineCache<u64>>>,
        retired: Arc<AtomicBool>,
    }
    impl std::fmt::Debug for OverlapAdmission {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("OverlapAdmission")
        }
    }
    struct RetireProbe {
        _claim: Arc<dyn RetainedHostAllocationClaim>,
        cache: std::sync::Weak<Mutex<PipelineCache<u64>>>,
        retired: Arc<AtomicBool>,
    }
    impl std::fmt::Debug for RetireProbe {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("RetireProbe")
        }
    }
    impl Drop for RetireProbe {
        fn drop(&mut self) {
            let cache = self.cache.upgrade().expect("live cache during requote");
            assert!(
                cache.try_lock().is_ok(),
                "obsolete claim retires outside cache lock"
            );
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    impl ProviderHostMetadataAdmission for OverlapAdmission {
        fn claim_view(
            &self,
            bytes: u64,
        ) -> Result<Box<dyn RetainedHostViewGrant>, ProviderHostMetadataError> {
            self.admission.claim_view(bytes)
        }
        fn claim_record(
            &self,
            bytes: u64,
        ) -> Result<Arc<dyn RetainedHostAllocationClaim>, ProviderHostMetadataError> {
            let claim = match self.failure {
                Some(error) => Err(error),
                None => self.admission.claim_record(bytes),
            };
            if self.first.swap(false, Ordering::SeqCst) {
                self.quoted.wait();
                self.release.wait();
                claim.map(|claim| {
                    Arc::new(RetireProbe {
                        _claim: claim,
                        cache: self.cache.clone(),
                        retired: self.retired.clone(),
                    }) as Arc<dyn RetainedHostAllocationClaim>
                })
            } else {
                claim
            }
        }
    }
    for same_key in [false, true] {
        for failure in [
            None,
            Some(ProviderHostMetadataError::ExceedsCapacity),
            Some(ProviderHostMetadataError::Cancelled),
            Some(ProviderHostMetadataError::StaleAuthority),
            Some(ProviderHostMetadataError::Overflow),
            Some(ProviderHostMetadataError::OwnerFailure),
        ] {
            let cache = Arc::new(Mutex::new(PipelineCache::<u64>::default()));
            let retired = Arc::new(AtomicBool::new(false));
            let overlap = OverlapAdmission {
                admission: Admission::default(),
                failure,
                first: AtomicBool::new(true),
                quoted: std::sync::Barrier::new(2),
                release: std::sync::Barrier::new(2),
                cache: Arc::downgrade(&cache),
                retired: retired.clone(),
            };
            let winner_admission = Admission::default();
            std::thread::scope(|scope| {
                let worker = scope.spawn(|| -> anyhow::Result<_> {
                    match PipelineCache::reserve(&cache, 100, Some(&overlap))? {
                        CacheLookup::Hit(hit) => Ok((hit, false)),
                        CacheLookup::Miss(reservation) => {
                            let (selected, loser, inserted) =
                                reservation.publish(PipelineOwner::unbounded(100))?;
                            assert!(loser.is_none());
                            Ok((selected, inserted))
                        }
                    }
                });
                overlap.quoted.wait();
                let winner_key = if same_key { 100 } else { 200 };
                let (winner, _, _) = miss(&cache, winner_key, &winner_admission)
                    .publish(PipelineOwner::unbounded(winner_key))
                    .unwrap();
                overlap.release.wait();
                let result = worker.join().unwrap();
                if let Some(error) =
                    failure.filter(|error| *error != ProviderHostMetadataError::ExceedsCapacity)
                {
                    let error_result = match result {
                        Err(error) => error,
                        Ok(_) => {
                            panic!("logical admission error cannot be replaced by a cache winner")
                        }
                    };
                    assert_eq!(
                        error_result.downcast_ref::<ProviderHostMetadataError>(),
                        Some(&error)
                    );
                    assert_eq!(cache.lock().unwrap().len(), 1);
                    return;
                }
                let (selected, inserted) = result.unwrap();
                assert_eq!(inserted, !same_key);
                if same_key {
                    assert!(selected.ptr_eq(&winner));
                }
                assert_eq!(*selected.as_ref(), 100);
                assert_eq!(cache.lock().unwrap().len(), if same_key { 1 } else { 2 });
            });
            assert_eq!(retired.load(Ordering::SeqCst), failure.is_none());
            assert_eq!(overlap.admission.live(), 0);
            drop(cache);
            assert_eq!(winner_admission.live(), 0);
        }
    }
    let cache = Mutex::new(PipelineCache::<u64>::default());
    let admission = Admission::default();
    let first = miss(&cache, 9, &admission);
    let second = miss(&cache, 9, &admission);
    let a = PipelineAllocation::prepare(Some(&admission))
        .unwrap()
        .allocate(11u64)
        .unwrap();
    let b = PipelineAllocation::prepare(Some(&admission))
        .unwrap()
        .allocate(22u64)
        .unwrap();
    let (winner, loser, inserted) = first.publish(a).unwrap();
    assert!(inserted && loser.is_none());
    let (selected, loser, inserted) = second.publish(b).unwrap();
    assert!(!inserted && selected.ptr_eq(&winner));
    assert_eq!(*loser.as_ref().unwrap().as_ref(), 22);
    drop(loser);
    let before_hit = admission.live();
    admission.0.deny.store(true, Ordering::SeqCst);
    let hit = match PipelineCache::reserve(&cache, 9, Some(&admission)).unwrap() {
        CacheLookup::Hit(hit) => hit,
        CacheLookup::Miss(_) => panic!("warm hit"),
    };
    assert!(hit.ptr_eq(&winner));
    assert_eq!(admission.live(), before_hit);
    drop(cache);
    let wrapper = PipelineAllocation::<u64>::requested_bytes().unwrap();
    assert_eq!(admission.live(), wrapper);
    drop(winner);
    drop(selected);
    assert_eq!(admission.live(), wrapper);
    drop(hit);
    assert_eq!(admission.live(), 0);

    struct Probe(Arc<Ledger>, Arc<AtomicBool>);
    impl Drop for Probe {
        fn drop(&mut self) {
            assert_eq!(
                self.0.live.load(Ordering::SeqCst),
                PipelineAllocation::<Self>::requested_bytes().unwrap()
            );
            self.1.store(true, Ordering::SeqCst);
        }
    }
    let destroyed = Arc::new(AtomicBool::new(false));
    admission.0.deny.store(false, Ordering::SeqCst);
    let owner = PipelineAllocation::prepare(Some(&admission))
        .unwrap()
        .allocate(Probe(admission.0.clone(), destroyed.clone()))
        .unwrap();
    let alias = owner.clone();
    drop(owner);
    assert!(!destroyed.load(Ordering::SeqCst));
    drop(alias);
    assert!(destroyed.load(Ordering::SeqCst));
    assert_eq!(admission.live(), 0);
}

#[test]
fn replacement_quotes_old_plus_new_backing_and_preserves_pipeline_identity() {
    let cache = Mutex::new(PipelineCache::<u64>::default());
    let admission = Admission::default();
    for key in 0..3 {
        let reservation = miss(&cache, key, &admission);
        let pipeline = PipelineOwner::unbounded(key);
        reservation.publish(pipeline).unwrap();
    }
    let old_bytes = admission.live();
    let old_alias = cache.lock().unwrap().get(&0).unwrap().clone();
    let reservation = miss(&cache, 3, &admission);
    let new_bytes = admission.live();
    assert!(new_bytes > old_bytes);
    assert!(admission.0.peak.load(Ordering::SeqCst) >= old_bytes + new_bytes);
    reservation.publish(PipelineOwner::unbounded(3)).unwrap();
    assert!(old_alias.ptr_eq(cache.lock().unwrap().get(&0).unwrap()));
    drop(cache);
    assert_eq!(admission.live(), 0);
}

#[test]
fn persistence_scratch_has_one_exact_fallible_claim_and_no_payload_copy() {
    use crate::backend::wgpu::cache::persist::PreparedPipelinePersistence;
    let admission = Admission::default();
    let dir = std::path::Path::new("never-created-cache-path");
    let prepared = PreparedPipelinePersistence::prepare(dir, 42, Some(&admission)).unwrap();
    assert_eq!(
        admission.live(),
        (dir.as_os_str().as_encoded_bytes().len() + std::path::MAIN_SEPARATOR_STR.len() + 21)
            as u64
    );
    drop(prepared);
    assert_eq!(admission.live(), 0);
    admission.0.deny.store(true, Ordering::SeqCst);
    assert!(PreparedPipelinePersistence::prepare(dir, 42, Some(&admission)).is_err());
    assert_eq!(admission.live(), 0);
}
