use runmat_package::ContentDigest;
use runmat_package_cache::backend::conformance::MemoryBackend;
use runmat_package_cache::lease;
use runmat_package_cache::{
    acquire_lease, execute_gc, release_lease, renew_lease, BlobMetadata, CacheBackend, CacheObject,
    CacheState, CacheTransaction, CommitOutcome, GcPlan, GcPolicy, LeaseId, LeaseOwner,
    ObjectWrite, Pin, PinId,
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

#[test]
fn transactional_lease_lifecycle_is_backend_neutral_and_generation_checked() {
    futures::executor::block_on(async {
        let backend = MemoryBackend::new();
        let bytes = b"leased".to_vec();
        let metadata = BlobMetadata::from_bytes(&bytes);
        let digest = metadata.digest.clone();
        let mut state = CacheState::default();
        state
            .objects
            .insert(digest.clone(), CacheObject::Blob(metadata.clone()));
        let mut publish = CacheTransaction::metadata_only(0, state);
        publish.writes.insert(
            digest.clone(),
            ObjectWrite::new(CacheObject::Blob(metadata), Some(bytes)).unwrap(),
        );
        assert!(matches!(
            backend.commit(publish).await.unwrap(),
            CommitOutcome::Committed(_)
        ));

        let lease = acquire_lease(
            &backend,
            LeaseId::new("session").unwrap(),
            LeaseOwner::new("worker").unwrap(),
            BTreeSet::from([digest]),
            10,
            100,
            4,
        )
        .await
        .unwrap();
        let renewed = renew_lease(&backend, &lease, 20, 100, 4).await.unwrap();
        assert_eq!(renewed.expires_at_ms, 120);
        let mut stale = lease;
        stale.generation += 1;
        assert!(release_lease(&backend, &stale, 4).await.is_err());
        release_lease(&backend, &renewed, 4).await.unwrap();
        assert!(backend.snapshot().await.unwrap().state.leases.is_empty());
    });
}

#[test]
fn gc_atomically_expires_stale_leases_before_collecting_their_closure() {
    futures::executor::block_on(async {
        let backend = MemoryBackend::new();
        let bytes = b"expired".to_vec();
        let metadata = BlobMetadata::from_bytes(&bytes);
        let digest = metadata.digest.clone();
        let mut state = CacheState::default();
        state
            .objects
            .insert(digest.clone(), CacheObject::Blob(metadata.clone()));
        lease::acquire(
            &mut state,
            LeaseId::new("expired").unwrap(),
            LeaseOwner::new("worker").unwrap(),
            BTreeSet::from([digest.clone()]),
            10,
            10,
        )
        .unwrap();
        let mut publish = CacheTransaction::metadata_only(0, state);
        publish.writes.insert(
            digest.clone(),
            ObjectWrite::new(CacheObject::Blob(metadata), Some(bytes)).unwrap(),
        );
        backend.commit(publish).await.unwrap();

        let plan = execute_gc(&backend, GcPolicy::reclaim_to(21, u64::MAX), 4)
            .await
            .unwrap();
        assert!(plan.delete.contains(&digest));
        let state = backend.snapshot().await.unwrap().state;
        assert!(state.leases.is_empty());
        assert!(state.objects.is_empty());
    });
}
