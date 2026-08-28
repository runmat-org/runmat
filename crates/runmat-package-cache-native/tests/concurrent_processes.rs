use futures::executor::block_on;
use runmat_package_cache::{
    BlobMetadata, CacheBackend, CacheObject, CacheTransaction, CommitOutcome, ObjectWrite,
};
use runmat_package_cache_native::SqliteCacheBackend;

fn blob_transaction(
    revision: u64,
    mut state: runmat_package_cache::CacheState,
    bytes: &[u8],
) -> CacheTransaction {
    let metadata = BlobMetadata::from_bytes(bytes);
    let digest = metadata.digest.clone();
    state
        .objects
        .insert(digest.clone(), CacheObject::Blob(metadata.clone()));
    let mut transaction = CacheTransaction::metadata_only(revision, state);
    transaction.writes.insert(
        digest,
        ObjectWrite::new(CacheObject::Blob(metadata), Some(bytes.to_vec())).unwrap(),
    );
    transaction
}

#[test]
fn independent_connections_serialize_stale_writers() {
    block_on(async {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.sqlite3");
        let first = SqliteCacheBackend::open_path(&path, None).unwrap();
        let second = SqliteCacheBackend::open_path(&path, None).unwrap();
        let left = first.snapshot().await.unwrap();
        let right = second.snapshot().await.unwrap();

        assert!(matches!(
            first
                .commit(blob_transaction(left.revision, left.state, b"left"))
                .await
                .unwrap(),
            CommitOutcome::Committed(_)
        ));
        assert!(matches!(
            second
                .commit(blob_transaction(right.revision, right.state, b"right"))
                .await
                .unwrap(),
            CommitOutcome::Conflict { actual_revision: 1 }
        ));
        assert_eq!(second.snapshot().await.unwrap().revision, 1);
    });
}
