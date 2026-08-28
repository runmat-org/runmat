use futures::executor::block_on;
use runmat_package::{ContentDigest, NormalizedRelativePath};
use runmat_package_cache::backend::conformance::MemoryBackend;
use runmat_package_cache::{
    cache_server_project_snapshot, load_server_project_snapshot, ArchiveLimits, CacheBackend,
    CommitOutcome, ServerProjectTreeInventory, SnapshotBlob, TreeEntry, TreeInventoryEntry,
    TreeManifest,
};

fn inventory() -> ServerProjectTreeInventory {
    let manifest = b"[package]\nname = \"helper\"\n".to_vec();
    let source = b"function y = helper(); y = 42; end\n".to_vec();
    let manifest_blob = SnapshotBlob::new(manifest.clone());
    let source_blob = SnapshotBlob::new(source.clone());
    let tree = TreeManifest::new(vec![
        TreeEntry::file(
            NormalizedRelativePath::new("runmat.toml").unwrap(),
            manifest_blob.digest,
            manifest.len() as u64,
            false,
        ),
        TreeEntry::directory(NormalizedRelativePath::new("src").unwrap()),
        TreeEntry::file(
            NormalizedRelativePath::new("src/helper.m").unwrap(),
            source_blob.digest,
            source.len() as u64,
            false,
        ),
    ])
    .unwrap();
    ServerProjectTreeInventory {
        project: "proj_0123456789abcdef0123456789abcdef".to_string(),
        snapshot: "snap_0123456789abcdef0123456789abcdef".to_string(),
        tree_digest: tree.digest,
        entries: vec![
            TreeInventoryEntry::file("runmat.toml", manifest, false),
            TreeInventoryEntry::directory("src"),
            TreeInventoryEntry::file("src/helper.m", source, false),
        ],
    }
}

#[test]
fn server_inventory_is_verified_and_round_trips_through_the_shared_cache() {
    let snapshot = inventory()
        .into_snapshot("https://api.runmat.com", ArchiveLimits::default())
        .unwrap();
    let backend = MemoryBackend::new();
    block_on(async {
        let current = backend.snapshot().await.unwrap();
        let transaction =
            cache_server_project_snapshot(current.revision, current.state, &snapshot, 1).unwrap();
        assert!(matches!(
            backend.commit(transaction).await.unwrap(),
            CommitOutcome::Committed(_)
        ));
        let loaded = load_server_project_snapshot(&backend, snapshot.source.clone())
            .await
            .unwrap();
        assert_eq!(loaded, snapshot);
    });
}

#[test]
fn server_inventory_rejects_server_digest_tampering_and_binds_service_identity() {
    let inventory = inventory();
    let first = inventory
        .clone()
        .into_snapshot("https://api.runmat.com", ArchiveLimits::default())
        .unwrap();
    let second = inventory
        .clone()
        .into_snapshot("https://other.runmat.example", ArchiveLimits::default())
        .unwrap();
    assert_ne!(first.source, second.source);

    let mut tampered = inventory;
    tampered.tree_digest = ContentDigest::sha256(b"not the tree");
    assert!(tampered
        .into_snapshot("https://api.runmat.com", ArchiveLimits::default())
        .is_err());
}

#[test]
fn server_inventory_digest_matches_the_server_golden_fixture() {
    let bytes = b"[package]\nname='one'\n".to_vec();
    let blob = SnapshotBlob::new(bytes.clone());
    let tree = TreeManifest::new(vec![TreeEntry::file(
        NormalizedRelativePath::new("runmat.toml").unwrap(),
        blob.digest,
        bytes.len() as u64,
        false,
    )])
    .unwrap();
    assert_eq!(
        tree.digest.to_string(),
        "sha256:a07604193face0cbad7ade6c616df22b6560c48f007c88b5c3b2cbab75ae8e32"
    );
    ServerProjectTreeInventory {
        project: "proj_0123456789abcdef0123456789abcdef".to_string(),
        snapshot: "snap_0123456789abcdef0123456789abcdef".to_string(),
        tree_digest: tree.digest,
        entries: vec![TreeInventoryEntry::file("runmat.toml", bytes, false)],
    }
    .into_snapshot("https://api.runmat.com", ArchiveLimits::default())
    .unwrap();
}
