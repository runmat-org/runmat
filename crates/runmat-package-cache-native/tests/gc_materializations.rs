use futures::executor::block_on;
use runmat_package::NormalizedRelativePath;
use runmat_package_cache::{
    BlobMetadata, CacheBackend, CacheObject, CacheTransaction, GcPolicy, ObjectWrite, TreeEntry,
    TreeManifest,
};
use runmat_package_cache_native::filesystem::CacheLayout;
use runmat_package_cache_native::materialize::materialize_tree;
use runmat_package_cache_native::{gc, NativeCacheLease, SqliteCacheBackend};
use std::sync::Arc;

#[test]
fn active_session_lease_protects_payload_and_physical_tree_until_drop() {
    block_on(async {
        let directory = tempfile::tempdir().unwrap();
        let layout = CacheLayout::new(directory.path().join("cache"));
        layout.create().unwrap();
        let backend = Arc::new(SqliteCacheBackend::open_path(&layout.database, None).unwrap());
        let bytes = b"function y = helper(); y = 1; end\n".to_vec();
        let blob = BlobMetadata::from_bytes(&bytes);
        let tree = TreeManifest::new(vec![TreeEntry::file(
            NormalizedRelativePath::new("helper.m").unwrap(),
            blob.digest.clone(),
            bytes.len() as u64,
            false,
        )])
        .unwrap();
        let mut state = backend.snapshot().await.unwrap().state;
        state
            .objects
            .insert(blob.digest.clone(), CacheObject::Blob(blob.clone()));
        state
            .objects
            .insert(tree.digest.clone(), CacheObject::Tree(tree.clone()));
        let mut publish = CacheTransaction::metadata_only(0, state);
        publish.writes.insert(
            blob.digest.clone(),
            ObjectWrite::new(CacheObject::Blob(blob), Some(bytes)).unwrap(),
        );
        backend.commit(publish).await.unwrap();
        let mounted = materialize_tree(&backend, &layout, &tree).await.unwrap();
        let lease =
            NativeCacheLease::acquire(backend.clone(), [tree.digest.clone()].into_iter().collect())
                .await
                .unwrap()
                .unwrap();

        let protected = gc::execute(
            &backend,
            &layout,
            GcPolicy::reclaim_to(now_ms(), u64::MAX),
            4,
        )
        .await
        .unwrap();
        assert!(!protected.delete.contains(&tree.digest));
        assert!(mounted.exists());

        drop(lease);
        let collected = gc::execute(
            &backend,
            &layout,
            GcPolicy::reclaim_to(now_ms(), u64::MAX),
            4,
        )
        .await
        .unwrap();
        assert!(collected.delete.contains(&tree.digest));
        assert!(!mounted.exists());
    });
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[test]
fn orphaned_digest_named_tree_is_recovered_without_touching_unknown_entries() {
    block_on(async {
        let directory = tempfile::tempdir().unwrap();
        let layout = CacheLayout::new(directory.path().join("cache"));
        layout.create().unwrap();
        let backend = SqliteCacheBackend::open_path(&layout.database, None).unwrap();
        let orphan = runmat_package::ContentDigest::sha256("orphan");
        let orphan_path = layout.tree_path(&orphan);
        std::fs::create_dir_all(&orphan_path).unwrap();
        std::fs::write(orphan_path.join("file"), b"orphan").unwrap();
        let unknown = layout.trees.join("operator-note");
        std::fs::create_dir_all(&unknown).unwrap();

        assert_eq!(
            gc::remove_orphaned_trees(&backend, &layout).await.unwrap(),
            vec![orphan]
        );
        assert!(!orphan_path.exists());
        assert!(unknown.exists());
    });
}
