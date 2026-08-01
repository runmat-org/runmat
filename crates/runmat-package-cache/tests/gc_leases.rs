use runmat_package::ContentDigest;
use runmat_package_cache::lease;
use runmat_package_cache::{
    BlobMetadata, CacheObject, CacheState, GcPlan, GcPolicy, LeaseId, LeaseOwner, Pin, PinId,
};
use std::collections::BTreeSet;

fn add_blob(state: &mut CacheState, bytes: &[u8]) -> ContentDigest {
    let metadata = BlobMetadata::from_bytes(bytes);
    let digest = metadata.digest.clone();
    state
        .objects
        .insert(digest.clone(), CacheObject::Blob(metadata));
    digest
}

#[test]
fn active_leases_and_pins_are_gc_roots() {
    let mut state = CacheState::default();
    let leased = add_blob(&mut state, b"leased");
    let pinned = add_blob(&mut state, b"pinned");
    let garbage = add_blob(&mut state, b"garbage");
    lease::acquire(
        &mut state,
        LeaseId::new("worker-lease").unwrap(),
        LeaseOwner::new("worker").unwrap(),
        BTreeSet::from([leased.clone()]),
        10,
        100,
    )
    .unwrap();
    state.pins.insert(
        PinId::new("frozen-graph").unwrap(),
        Pin {
            id: PinId::new("frozen-graph").unwrap(),
            objects: BTreeSet::from([pinned.clone()]),
            reason: "installed graph".to_string(),
            created_at_ms: 10,
        },
    );

    let plan = GcPlan::build(&state, GcPolicy::reclaim_to(20, u64::MAX));
    assert_eq!(plan.delete, BTreeSet::from([garbage]));
    assert!(!plan.delete.contains(&leased));
    assert!(!plan.delete.contains(&pinned));
}

#[test]
fn renewal_is_owner_and_generation_checked_and_expiry_releases_roots() {
    let mut state = CacheState::default();
    let digest = add_blob(&mut state, b"leased");
    let id = LeaseId::new("lease").unwrap();
    let owner = LeaseOwner::new("owner").unwrap();
    let lease = lease::acquire(
        &mut state,
        id.clone(),
        owner.clone(),
        BTreeSet::from([digest.clone()]),
        10,
        10,
    )
    .unwrap();
    assert!(lease::renew(&mut state, &id, &owner, lease.generation + 1, 11, 10).is_err());
    lease::renew(&mut state, &id, &owner, lease.generation, 11, 10).unwrap();
    assert!(lease::expire(&mut state, 20).is_empty());
    assert_eq!(lease::expire(&mut state, 21), vec![id]);
    assert_eq!(
        GcPlan::build(&state, GcPolicy::reclaim_to(21, u64::MAX)).delete,
        BTreeSet::from([digest])
    );
}
