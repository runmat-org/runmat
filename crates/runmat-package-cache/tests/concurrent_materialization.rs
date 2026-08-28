use runmat_package_cache::materialize;
use runmat_package_cache::{
    BlobMetadata, CacheObject, CacheState, LeaseId, LeaseOwner, MaterializationState,
};
use std::collections::BTreeSet;

#[test]
fn materialization_only_promotes_after_verification() {
    let mut state = CacheState::default();
    let metadata = BlobMetadata::from_bytes(b"tree payload");
    let digest = metadata.digest.clone();
    state
        .objects
        .insert(digest.clone(), CacheObject::Blob(metadata));
    let lease = LeaseId::new("materializer").unwrap();
    runmat_package_cache::lease::acquire(
        &mut state,
        lease.clone(),
        LeaseOwner::new("worker").unwrap(),
        BTreeSet::from([digest.clone()]),
        1,
        100,
    )
    .unwrap();

    materialize::begin(&mut state, &digest, lease, "attempt-1", 2).unwrap();
    assert!(materialize::promote(&mut state, &digest, 3).is_err());
    materialize::verify(&mut state, &digest, 3).unwrap();
    materialize::promote(&mut state, &digest, 4).unwrap();
    assert_eq!(
        state.materializations[&digest].state,
        MaterializationState::Promoted
    );
}

#[test]
fn recovery_drops_staging_and_missing_dependency_closure() {
    let mut state = CacheState::default();
    let blob = BlobMetadata::from_bytes(b"file");
    let missing = blob.digest.clone();
    state
        .objects
        .insert(missing.clone(), CacheObject::Blob(blob));
    let tree =
        runmat_package_cache::TreeManifest::new(vec![runmat_package_cache::TreeEntry::file(
            runmat_package::NormalizedRelativePath::new("src/file.m").unwrap(),
            missing.clone(),
            4,
            false,
        )])
        .unwrap();
    let tree_digest = tree.digest.clone();
    state
        .objects
        .insert(tree_digest.clone(), CacheObject::Tree(tree));

    let plan = runmat_package_cache::state::RecoveryPlan::inspect(&state, 0, [missing.clone()]);
    plan.apply(&mut state);
    assert!(!state.objects.contains_key(&missing));
    assert!(!state.objects.contains_key(&tree_digest));
    state.validate().unwrap();
}
