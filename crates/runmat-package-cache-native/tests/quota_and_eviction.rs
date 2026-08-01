use futures::executor::block_on;
use runmat_package_cache::{
    BackendError, BlobMetadata, CacheBackend, CacheObject, CacheTransaction, GcPolicy, ObjectWrite,
};
use runmat_package_cache_native::{gc, SqliteCacheBackend};

fn blob_transaction(
    revision: u64,
    mut state: runmat_package_cache::CacheState,
    bytes: &[u8],
) -> (runmat_package::ContentDigest, CacheTransaction) {
    let metadata = BlobMetadata::from_bytes(bytes);
    let digest = metadata.digest.clone();
    state
        .objects
        .insert(digest.clone(), CacheObject::Blob(metadata.clone()));
    let mut transaction = CacheTransaction::metadata_only(revision, state);
    transaction.writes.insert(
        digest.clone(),
        ObjectWrite::new(CacheObject::Blob(metadata), Some(bytes.to_vec())).unwrap(),
    );
    (digest, transaction)
}

#[test]
fn sqlite_quota_failure_rolls_back_state_and_payload() {
    block_on(async {
        let backend = SqliteCacheBackend::open_in_memory(Some(3)).unwrap();
        let (digest, transaction) =
            blob_transaction(0, backend.snapshot().await.unwrap().state, b"four");
        assert!(matches!(
            backend.commit(transaction).await,
            Err(BackendError::QuotaExceeded { .. })
        ));
        assert_eq!(backend.snapshot().await.unwrap().revision, 0);
        assert_eq!(backend.read_object_bytes(&digest).await.unwrap(), None);
    });
}

#[test]
fn native_gc_executes_portable_plan_atomically() {
    block_on(async {
        let backend = SqliteCacheBackend::open_in_memory(None).unwrap();
        let (digest, transaction) =
            blob_transaction(0, backend.snapshot().await.unwrap().state, b"garbage");
        backend.commit(transaction).await.unwrap();
        let plan = gc::execute(&backend, GcPolicy::reclaim_to(10, 1), 3)
            .await
            .unwrap();
        assert!(plan.delete.contains(&digest));
        assert_eq!(backend.snapshot().await.unwrap().revision, 2);
        assert_eq!(backend.read_object_bytes(&digest).await.unwrap(), None);
    });
}
