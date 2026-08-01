use runmat_package::{GitCommitId, GitSourceId, NormalizedRelativePath};
use runmat_package_cache::{
    ArchiveLimits, GitInventoryEntry, GitSnapshot, GitTreeInventory, SnapshotBlob, TreeEntry,
    TreeEntryKind, TreeManifest,
};

#[test]
fn git_snapshot_has_compact_strict_verified_wire_form() {
    let blob = SnapshotBlob::new(b"answer = 42;\n".to_vec());
    let tree = TreeManifest::new(vec![TreeEntry::file(
        NormalizedRelativePath::new("main.m").unwrap(),
        blob.digest.clone(),
        blob.bytes.len() as u64,
        false,
    )])
    .unwrap();
    let source = GitSourceId::new(
        "https://example.com/acme/project.git",
        "0123456789abcdef0123456789abcdef01234567"
            .parse::<GitCommitId>()
            .unwrap(),
        NormalizedRelativePath::new(".").unwrap(),
        tree.digest.clone(),
    )
    .unwrap();
    let snapshot = GitSnapshot::new(source, tree, vec![blob]).unwrap();
    let json = serde_json::to_string(&snapshot).unwrap();
    assert!(json.contains("YW5zd2VyID0gNDI7Cg=="));
    assert!(!json.contains("[97,110,115,119,101,114"));
    assert_eq!(
        serde_json::from_str::<GitSnapshot>(&json).unwrap(),
        snapshot
    );
}

#[test]
fn git_snapshot_rejects_missing_extra_and_tampered_blobs() {
    let blob = SnapshotBlob::new(b"file".to_vec());
    let tree = TreeManifest::new(vec![TreeEntry::file(
        NormalizedRelativePath::new("main.m").unwrap(),
        blob.digest.clone(),
        4,
        false,
    )])
    .unwrap();
    let source = GitSourceId::new(
        "https://example.com/acme/project.git",
        "0123456789abcdef0123456789abcdef01234567"
            .parse::<GitCommitId>()
            .unwrap(),
        NormalizedRelativePath::new(".").unwrap(),
        tree.digest.clone(),
    )
    .unwrap();
    assert!(GitSnapshot::new(source.clone(), tree.clone(), vec![]).is_err());
    assert!(GitSnapshot::new(
        source.clone(),
        tree.clone(),
        vec![blob.clone(), SnapshotBlob::new(b"extra".to_vec())]
    )
    .is_err());
    let mut tampered = blob;
    tampered.bytes[0] ^= 1;
    assert!(GitSnapshot::new(source, tree, vec![tampered]).is_err());
}

#[test]
fn gateway_inventory_builds_the_same_canonical_snapshot() {
    let inventory = GitTreeInventory {
        commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
        entries: vec![
            GitInventoryEntry::file("src/main.m", b"answer = 42;\n".to_vec(), true),
            GitInventoryEntry::symlink("src/alias.m", "main.m"),
            GitInventoryEntry::directory("src"),
        ],
    };
    let snapshot = inventory
        .into_snapshot(
            "https://example.com/acme/project.git",
            "package",
            ArchiveLimits::default(),
        )
        .unwrap();
    assert_eq!(snapshot.source.subdir.as_str(), "package");
    assert_eq!(snapshot.tree.file_count, 1);
    assert!(snapshot
        .tree
        .entries
        .iter()
        .any(|entry| entry.kind == TreeEntryKind::Symlink));
    snapshot.validate().unwrap();
}

#[test]
fn gateway_inventory_rejects_unsafe_or_inconsistent_entries() {
    let commit = "0123456789abcdef0123456789abcdef01234567".to_string();
    let build = |entries| {
        GitTreeInventory {
            commit: commit.clone(),
            entries,
        }
        .into_snapshot(
            "https://example.com/acme/project.git",
            ".",
            ArchiveLimits::default(),
        )
    };

    assert!(build(vec![GitInventoryEntry::file(
        "../escape.m",
        Vec::new(),
        false
    )])
    .is_err());
    assert!(build(vec![
        GitInventoryEntry::file("Main.m", Vec::new(), false),
        GitInventoryEntry::file("main.m", Vec::new(), false),
    ])
    .is_err());
    assert!(build(vec![GitInventoryEntry::symlink("link.m", "../../escape.m")]).is_err());
}

#[test]
fn server_gateway_wire_contract_deserializes_into_portable_authority() {
    let wire = r#"{
        "commit":"0123456789abcdef0123456789abcdef01234567",
        "entries":[
            {"path":"src","kind":"directory","executable":false},
            {"path":"src/main.m","kind":"file","bytes":"YW5zd2VyID0gNDI7Cg==","executable":false}
        ]
    }"#;
    let inventory: GitTreeInventory = serde_json::from_str(wire).unwrap();
    let snapshot = inventory
        .into_snapshot(
            "https://github.com/runmat-org/runmat",
            ".",
            ArchiveLimits::default(),
        )
        .unwrap();
    assert_eq!(snapshot.blobs[0].bytes, b"answer = 42;\n");
    snapshot.validate().unwrap();
}
