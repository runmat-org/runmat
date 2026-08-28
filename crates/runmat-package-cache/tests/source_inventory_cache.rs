use futures::executor::block_on;
use runmat_package::{
    ContentDigest, NormalizedRelativePath, SourceInventory, SourceInventoryEntry,
    SOURCE_INVENTORY_SCHEMA_VERSION,
};
use runmat_package_cache::backend::conformance::MemoryBackend;
use runmat_package_cache::{
    cache_git_snapshot, cache_source_inventory, load_source_inventory, ArchiveLimits, CacheBackend,
    GitInventoryEntry, GitTreeInventory,
};

#[test]
fn source_inventory_is_cached_by_tree_and_schema_with_verified_bytes() {
    block_on(async {
        let backend = MemoryBackend::new();
        let git = GitTreeInventory {
            commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            entries: vec![GitInventoryEntry::file(
                "src/main.m",
                b"answer = 42;\n".to_vec(),
                false,
            )],
        }
        .into_snapshot(
            "https://example.com/acme/project.git",
            ".",
            ArchiveLimits::default(),
        )
        .unwrap();
        let initial = backend.snapshot().await.unwrap();
        backend
            .commit(cache_git_snapshot(initial.revision, initial.state, &git, 1).unwrap())
            .await
            .unwrap();
        let inventory = SourceInventory {
            schema_version: SOURCE_INVENTORY_SCHEMA_VERSION,
            tree_digest: git.tree.digest.clone(),
            entries: vec![SourceInventoryEntry {
                source_root: NormalizedRelativePath::new("src").unwrap(),
                relative_path: NormalizedRelativePath::new("main.m").unwrap(),
                qualified_name: "main".to_string(),
                package_path: None,
                class_name: None,
                class_qualified_name: None,
                is_private: false,
            }],
            package_dirs: Vec::new(),
            class_dirs: Vec::new(),
            private_dirs: Vec::new(),
        };
        let snapshot = backend.snapshot().await.unwrap();
        backend
            .commit(
                cache_source_inventory(snapshot.revision, snapshot.state, &inventory, 2).unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            load_source_inventory(&backend, &git.tree.digest, SOURCE_INVENTORY_SCHEMA_VERSION)
                .await
                .unwrap(),
            inventory
        );
        assert!(
            load_source_inventory(&backend, &ContentDigest::sha256("missing"), 1)
                .await
                .is_err()
        );
    });
}
