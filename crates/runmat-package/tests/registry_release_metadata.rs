use runmat_package::{
    CanonicalPackageId, ContentDigest, DependencyGroup, PackageVersion, RegistryId, RegistryOrigin,
    RegistryPackageReference, RegistryReleaseDependency, RegistryReleaseId,
    RegistryReleaseMetadata, RegistrySourceId,
};

fn golden_metadata() -> RegistryReleaseMetadata {
    RegistryReleaseMetadata {
        singleton: true,
        runmat_requirement: Some("^0.6".to_string()),
        dependencies: vec![RegistryReleaseDependency {
            alias: "math".to_string(),
            package: RegistryPackageReference {
                registry: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
                namespace: "acme".to_string(),
                name: "math".to_string(),
            },
            requirement: "^2".to_string(),
            group: DependencyGroup::Runtime,
            target: Some("wasm32-unknown-unknown".to_string()),
            optional: false,
            default_features: true,
            features: vec!["sparse".to_string()],
        }],
        features: [("default".to_string(), vec!["math/sparse".to_string()])]
            .into_iter()
            .collect(),
        required_capabilities: vec!["filesystem".to_string()],
        optional_capabilities: vec!["gpu".to_string()],
        readme_digest: Some(format!("sha256:{}", "c".repeat(64))),
        license: Some("MIT".to_string()),
    }
}

#[test]
fn release_digest_matches_the_server_cross_host_golden() {
    let mut source = RegistrySourceId {
        registry_origin: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
        package: CanonicalPackageId::new(RegistryId::default(), "acme", "tools").unwrap(),
        release: RegistryReleaseId::new("rel_0123456789abcdef0123456789abcdef").unwrap(),
        version: "1.2.3".parse::<PackageVersion>().unwrap(),
        release_digest: ContentDigest::sha256([]),
        artifact_digest: format!("sha256:{}", "a".repeat(64)).parse().unwrap(),
        tree_digest: format!("sha256:{}", "b".repeat(64)).parse().unwrap(),
    };
    let metadata = golden_metadata();
    let digest = metadata.compute_digest(&source).unwrap();
    assert_eq!(
        digest.to_string(),
        "sha256:018ec4b3b5d0f26c93bda8809cca0c3f0aca03d6171cf432b8ef916198212b75"
    );
    source.release_digest = digest;
    metadata.validate_source(&source).unwrap();
}

#[test]
fn release_metadata_tampering_is_detected() {
    let mut source = RegistrySourceId {
        registry_origin: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
        package: CanonicalPackageId::new(RegistryId::default(), "acme", "tools").unwrap(),
        release: RegistryReleaseId::new("rel_0123456789abcdef0123456789abcdef").unwrap(),
        version: "1.2.3".parse::<PackageVersion>().unwrap(),
        release_digest: ContentDigest::sha256([]),
        artifact_digest: format!("sha256:{}", "a".repeat(64)).parse().unwrap(),
        tree_digest: format!("sha256:{}", "b".repeat(64)).parse().unwrap(),
    };
    let metadata = golden_metadata();
    source.release_digest = metadata.compute_digest(&source).unwrap();
    let mut changed = metadata;
    changed.required_capabilities.push("network".to_string());
    assert!(changed.validate_source(&source).is_err());
}
