use futures::executor::block_on;
use runmat_package_cache::{
    BlobMetadata, CacheBackend, CacheObject, CacheTransaction, ObjectWrite,
};
use runmat_package_cache_native::SqliteCacheBackend;

#[test]
fn committed_state_and_payload_survive_reopen_together() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cache.sqlite3");
    let digest = block_on(async {
        let backend = SqliteCacheBackend::open_path(&path, None).unwrap();
        let bytes = b"durable".to_vec();
        let metadata = BlobMetadata::from_bytes(&bytes);
        let digest = metadata.digest.clone();
        let mut state = backend.snapshot().await.unwrap().state;
        state
            .objects
            .insert(digest.clone(), CacheObject::Blob(metadata.clone()));
        let mut transaction = CacheTransaction::metadata_only(0, state);
        transaction.writes.insert(
            digest.clone(),
            ObjectWrite::new(CacheObject::Blob(metadata), Some(bytes)).unwrap(),
        );
        backend.commit(transaction).await.unwrap();
        digest
    });

    block_on(async {
        let reopened = SqliteCacheBackend::open_path(&path, None).unwrap();
        assert_eq!(reopened.snapshot().await.unwrap().revision, 1);
        assert_eq!(
            reopened.read_object_bytes(&digest).await.unwrap(),
            Some(b"durable".to_vec())
        );
    });
}

#[test]
fn externally_missing_payload_is_reported_as_corruption() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cache.sqlite3");
    let digest = block_on(async {
        let backend = SqliteCacheBackend::open_path(&path, None).unwrap();
        let bytes = b"evicted".to_vec();
        let metadata = BlobMetadata::from_bytes(&bytes);
        let digest = metadata.digest.clone();
        let mut state = backend.snapshot().await.unwrap().state;
        state
            .objects
            .insert(digest.clone(), CacheObject::Blob(metadata.clone()));
        let mut transaction = CacheTransaction::metadata_only(0, state);
        transaction.writes.insert(
            digest.clone(),
            ObjectWrite::new(CacheObject::Blob(metadata), Some(bytes)).unwrap(),
        );
        backend.commit(transaction).await.unwrap();
        digest
    });
    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute(
            "DELETE FROM object_payloads WHERE digest = ?1",
            [digest.to_string()],
        )
        .unwrap();
    drop(connection);

    block_on(async {
        let reopened = SqliteCacheBackend::open_path(&path, None).unwrap();
        assert!(reopened.snapshot().await.is_err());
    });
}
