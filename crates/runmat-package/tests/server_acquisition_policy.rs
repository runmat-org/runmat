use runmat_package::{
    plan_server_project_acquisition, validate_server_project_acquisition, ContentDigest,
    ServerProjectSourceId, ServerSnapshotSelector, SourceAcquisitionIntent,
    SourceAcquisitionPolicy, SourceLockAction,
};

fn locked() -> ServerProjectSourceId {
    ServerProjectSourceId::new(
        "https://api.runmat.com/",
        "proj_0123456789abcdef0123456789abcdef",
        "snap_0123456789abcdef0123456789abcdef",
        ContentDigest::sha256(b"tree"),
    )
    .unwrap()
}

#[test]
fn mutable_server_tags_resolve_only_when_lock_policy_allows_it() {
    let selector = ServerSnapshotSelector::from_manifest(Some("main")).unwrap();
    let plan = plan_server_project_acquisition(
        "https://api.runmat.com",
        "proj_0123456789abcdef0123456789abcdef",
        selector.clone(),
        None,
        SourceAcquisitionIntent::Execute,
        SourceAcquisitionPolicy::default(),
    )
    .unwrap();
    assert_eq!(plan.selector, selector);
    assert_eq!(plan.lock_action, SourceLockAction::Write);
    assert!(plan.expected.is_none());

    let locked = locked();
    let replay = plan_server_project_acquisition(
        "https://api.runmat.com/",
        &locked.project,
        selector,
        Some(&locked),
        SourceAcquisitionIntent::Execute,
        SourceAcquisitionPolicy {
            locked: true,
            frozen: false,
            offline: true,
        },
    )
    .unwrap();
    assert_eq!(
        replay.selector,
        ServerSnapshotSelector::Exact {
            value: locked.snapshot.clone()
        }
    );
    assert_eq!(replay.expected.as_ref(), Some(&locked));
    assert!(!replay.allow_network);
    assert_eq!(replay.lock_action, SourceLockAction::Preserve);
}

#[test]
fn server_policy_rejects_missing_or_cross_server_locks_and_validates_exact_results() {
    let locked = locked();
    assert!(plan_server_project_acquisition(
        "https://api.runmat.com",
        &locked.project,
        ServerSnapshotSelector::from_manifest(None).unwrap(),
        None,
        SourceAcquisitionIntent::Execute,
        SourceAcquisitionPolicy {
            locked: true,
            frozen: false,
            offline: false,
        },
    )
    .is_err());
    assert!(plan_server_project_acquisition(
        "https://other.runmat.example",
        &locked.project,
        ServerSnapshotSelector::from_manifest(None).unwrap(),
        Some(&locked),
        SourceAcquisitionIntent::Execute,
        SourceAcquisitionPolicy::default(),
    )
    .is_err());

    let plan = plan_server_project_acquisition(
        "https://api.runmat.com",
        &locked.project,
        ServerSnapshotSelector::from_manifest(None).unwrap(),
        Some(&locked),
        SourceAcquisitionIntent::Execute,
        SourceAcquisitionPolicy::default(),
    )
    .unwrap();
    validate_server_project_acquisition(&plan, &locked).unwrap();
    let moved = ServerProjectSourceId {
        snapshot: "snap_ffffffffffffffffffffffffffffffff".to_string(),
        ..locked
    };
    assert!(validate_server_project_acquisition(&plan, &moved).is_err());
}

#[test]
fn explicit_snapshot_ids_are_exact_even_without_an_existing_lock() {
    let snapshot = "snap_0123456789abcdef0123456789abcdef";
    let selector = ServerSnapshotSelector::from_manifest(Some(snapshot)).unwrap();
    assert_eq!(
        selector,
        ServerSnapshotSelector::Exact {
            value: snapshot.to_string()
        }
    );
}

#[test]
fn malformed_snapshot_ids_are_not_reinterpreted_as_tags() {
    for selector in [
        "snap_",
        "snap_0123",
        "snap_0123456789ABCDEF0123456789ABCDEF",
        "snap_gggggggggggggggggggggggggggggggg",
    ] {
        assert!(ServerSnapshotSelector::from_manifest(Some(selector)).is_err());
    }
}
