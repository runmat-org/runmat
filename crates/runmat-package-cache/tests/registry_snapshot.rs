use runmat_package::{
    CanonicalPackageId, ContentDigest, PackageVersion, RegistryId, RegistryOrigin,
    RegistryReleaseId, RegistrySourceId,
};
use runmat_package_cache::{
    ArchiveLimits, GitTreeInventory, RegistryArtifactInventory, TreeInventoryEntry,
    REGISTRY_ARTIFACT_SCHEMA_VERSION,
};

#[test]
fn registry_artifact_validates_artifact_and_tree_digests() {
    let inventory = RegistryArtifactInventory {
        schema_version: REGISTRY_ARTIFACT_SCHEMA_VERSION,
        entries: vec![
            TreeInventoryEntry::directory("src"),
            TreeInventoryEntry::file(
                "runmat.toml",
                b"[package]\nname = \"tools\"\n".to_vec(),
                false,
            ),
            TreeInventoryEntry::file("src/main.m", b"answer = 42;\n".to_vec(), false),
        ],
    };
    let bytes = inventory.canonical_bytes().unwrap();
    let tree = GitTreeInventory {
        commit: "a".repeat(40),
        entries: inventory.entries.clone(),
    }
    .into_snapshot(
        "https://example.test/tools.git",
        ".",
        ArchiveLimits::default(),
    )
    .unwrap()
    .tree;
    let source = RegistrySourceId {
        registry_origin: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
        package: CanonicalPackageId::new(RegistryId::default(), "acme", "tools").unwrap(),
        release: RegistryReleaseId::new("rel_0123456789abcdef0123456789abcdef").unwrap(),
        version: "1.2.3".parse::<PackageVersion>().unwrap(),
        release_digest: ContentDigest::sha256(b"release"),
        artifact_digest: ContentDigest::sha256(&bytes),
        tree_digest: tree.digest,
    };
    let snapshot = RegistryArtifactInventory::decode_snapshot(
        &bytes,
        source.clone(),
        ArchiveLimits::default(),
    )
    .unwrap();
    assert_eq!(snapshot.source, source);

    let mut corrupt = bytes;
    corrupt.push(b' ');
    assert!(RegistryArtifactInventory::decode_snapshot(
        &corrupt,
        snapshot.source,
        ArchiveLimits::default()
    )
    .is_err());
}
