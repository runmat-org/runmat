use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};
use chrono::TimeZone as _;
use ed25519_dalek::{Signer as _, SigningKey};
use runmat_package::{
    BuildProvenance, CanonicalPackageId, ContentDigest, DependencyGroup, EncryptedArtifactMetadata,
    PackageTrustTier, PackageVersion, RegistryId, RegistryOrigin, RegistryPackageReference,
    RegistryReleaseDependency, RegistryReleaseId, RegistryReleaseMetadata,
    RegistryReleaseSupplyChain, RegistrySourceId, RELEASE_SUPPLY_CHAIN_SCHEMA_VERSION,
};
use sha2::Digest as _;

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
        encryption: None,
        supply_chain: None,
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

#[test]
fn encrypted_release_digest_has_a_cross_host_v3_encoding() {
    let mut source = RegistrySourceId {
        registry_origin: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
        package: CanonicalPackageId::new(RegistryId::default(), "acme", "private").unwrap(),
        release: RegistryReleaseId::new("rel_99999999999999999999999999999999").unwrap(),
        version: "1.2.3".parse::<PackageVersion>().unwrap(),
        release_digest: ContentDigest::sha256([]),
        artifact_digest: ContentDigest::sha256("ciphertext"),
        tree_digest: ContentDigest::sha256("tree"),
    };
    let mut metadata = golden_metadata();
    metadata.encryption =
        Some(EncryptedArtifactMetadata::new(9, b"plaintext", [7; 12], &source).unwrap());
    source.release_digest = metadata.compute_digest(&source).unwrap();

    assert_eq!(
        source.release_digest.to_string(),
        "sha256:c0198cc057aa54fe9a34472e45a653d9262218114b9f4db70d286f8f99eb1fab"
    );
    metadata.validate_source(&source).unwrap();
}

fn assert_signed_release_digest_and_signature_match_the_server_cross_host_golden() {
    let signing_key = SigningKey::from_bytes(&[13; 32]);
    let package_id = "pkg_22222222222222222222222222222222";
    let mut supply_chain = RegistryReleaseSupplyChain {
        schema_version: RELEASE_SUPPLY_CHAIN_SCHEMA_VERSION,
        publication_id: "pub_33333333333333333333333333333333".to_string(),
        publisher_id: "ptp_11111111111111111111111111111111".to_string(),
        publisher_name: "release workflow".to_string(),
        public_key: STANDARD_NO_PAD.encode(signing_key.verifying_key().as_bytes()),
        key_fingerprint: format!(
            "sha256:{:x}",
            sha2::Sha256::digest(signing_key.verifying_key().as_bytes())
        ),
        payload_digest: format!("sha256:{}", "4".repeat(64)),
        sequence: 42,
        signed_at: chrono::Utc
            .timestamp_opt(1_700_000_000, 123_456_000)
            .single()
            .unwrap(),
        trust_tier: PackageTrustTier::Community,
        provenance: BuildProvenance {
            source_repository: "https://github.com/runmat-org/example".to_string(),
            source_commit: "a".repeat(40),
            builder_id: "https://github.com/actions/runner".to_string(),
            workflow_ref: Some(".github/workflows/release.yml@main".to_string()),
            invocation_id: "run-42".to_string(),
            inventory_digest: format!("sha256:{}", "5".repeat(64)),
            release_manifest_digest: format!("sha256:{}", "6".repeat(64)),
            sbom: None,
            license: Some("MIT".to_string()),
            wrapper: None,
        },
        signature: String::new(),
    };
    supply_chain.signature = STANDARD_NO_PAD.encode(
        signing_key
            .sign(&supply_chain.canonical_signing_bytes(package_id).unwrap())
            .to_bytes(),
    );
    let mut metadata = RegistryReleaseMetadata {
        singleton: false,
        runmat_requirement: Some("^0.6".to_string()),
        dependencies: Vec::new(),
        features: Default::default(),
        required_capabilities: Vec::new(),
        optional_capabilities: Vec::new(),
        readme_digest: None,
        license: Some("MIT".to_string()),
        encryption: None,
        supply_chain: Some(supply_chain),
    };
    let mut source = RegistrySourceId {
        registry_origin: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
        package: CanonicalPackageId::new(RegistryId::default(), "acme", "tools").unwrap(),
        release: RegistryReleaseId::new("rel_88888888888888888888888888888888").unwrap(),
        version: "1.2.3".parse::<PackageVersion>().unwrap(),
        release_digest: ContentDigest::sha256([]),
        artifact_digest: format!("sha256:{}", "1".repeat(64)).parse().unwrap(),
        tree_digest: format!("sha256:{}", "2".repeat(64)).parse().unwrap(),
    };
    source.release_digest = metadata.compute_digest(&source).unwrap();
    assert_eq!(
        source.release_digest.to_string(),
        "sha256:000a930b5af7e3ab6d2f1f508b6b1e8444b5c42ce1d4adf7ed9bf3046dab8cff"
    );
    metadata.verify_supply_chain(package_id, &source).unwrap();
    metadata
        .supply_chain
        .as_mut()
        .unwrap()
        .provenance
        .source_commit = "b".repeat(40);
    assert!(metadata.verify_supply_chain(package_id, &source).is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn signed_release_digest_and_signature_match_the_server_cross_host_golden() {
    assert_signed_release_digest_and_signature_match_the_server_cross_host_golden();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn signed_release_digest_and_signature_match_the_server_cross_host_golden_in_wasm() {
    assert_signed_release_digest_and_signature_match_the_server_cross_host_golden();
}
