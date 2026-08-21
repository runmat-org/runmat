use runmat_package::{
    plan_registry_acquisition, validate_registry_acquisition, CanonicalPackageId, ContentDigest,
    PackageVersion, RegistryId, RegistryOrigin, RegistryPolicyError, RegistryReleaseId,
    RegistrySourceId, SourceAcquisitionIntent, SourceAcquisitionPolicy, SourceLockAction,
};

fn source(version: &str, salt: u8) -> RegistrySourceId {
    RegistrySourceId {
        registry_origin: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
        package: CanonicalPackageId::new(RegistryId::default(), "acme", "tools").unwrap(),
        release: RegistryReleaseId::new(format!("rel_{}", format!("{salt:02x}").repeat(16)))
            .unwrap(),
        version: version.parse::<PackageVersion>().unwrap(),
        release_digest: ContentDigest::sha256([salt, 1]),
        artifact_digest: ContentDigest::sha256([salt, 2]),
        tree_digest: ContentDigest::sha256([salt, 3]),
    }
}

#[test]
fn locked_execute_preserves_exact_release_and_update_reselects() {
    let locked = source("1.2.3", 4);
    let execute = plan_registry_acquisition(
        "mirror".parse().unwrap(),
        "https://mirror.runmat.test/index/",
        locked.package.clone(),
        "^1".parse().unwrap(),
        Some(&locked),
        SourceAcquisitionIntent::Execute,
        SourceAcquisitionPolicy::default(),
    )
    .unwrap();
    assert_eq!(execute.expected.as_ref(), Some(&locked));
    assert_eq!(execute.lock_action, SourceLockAction::Preserve);
    assert_eq!(execute.index, "https://mirror.runmat.test/index");
    validate_registry_acquisition(&execute, &locked).unwrap();

    let update = plan_registry_acquisition(
        "mirror".parse().unwrap(),
        "https://mirror.runmat.test/index",
        locked.package.clone(),
        "^1".parse().unwrap(),
        Some(&locked),
        SourceAcquisitionIntent::Update,
        SourceAcquisitionPolicy::default(),
    )
    .unwrap();
    assert!(update.expected.is_none());
    assert_eq!(update.lock_action, SourceLockAction::Replace);
}

#[test]
fn frozen_and_locked_policy_fail_closed() {
    let locked = source("2.0.0", 7);
    let error = plan_registry_acquisition(
        RegistryId::default(),
        "https://packages.runmat.test",
        locked.package.clone(),
        "^1".parse().unwrap(),
        Some(&locked),
        SourceAcquisitionIntent::Execute,
        SourceAcquisitionPolicy::default(),
    )
    .unwrap_err();
    assert_eq!(error, RegistryPolicyError::LockVersionMismatch);

    let error = plan_registry_acquisition(
        RegistryId::default(),
        "https://packages.runmat.test",
        locked.package,
        "*".parse().unwrap(),
        None,
        SourceAcquisitionIntent::Execute,
        SourceAcquisitionPolicy {
            locked: true,
            ..SourceAcquisitionPolicy::default()
        },
    )
    .unwrap_err();
    assert_eq!(error, RegistryPolicyError::MissingLock);
}

#[test]
fn registry_identity_rejects_cross_authority_and_malformed_release_ids() {
    assert!(RegistryOrigin::new("http://packages.runmat.test").is_err());
    assert!(RegistryOrigin::new("https://packages.runmat.test/path").is_err());
    assert!(RegistryReleaseId::new("rel_not-hex").is_err());
}
