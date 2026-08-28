use futures::executor::block_on;
use runmat_package_cache::backend::conformance::MemoryBackend;
use runmat_package_cache::{
    BackendError, BlobMetadata, CacheBackend, CacheObject, CacheTransaction, CommitOutcome,
    ObjectWrite,
};

#[test]
fn quota_failure_and_stale_writer_publish_nothing() {
    block_on(async {
        let backend = MemoryBackend::with_quota(3);
        let initial = backend.snapshot().await.unwrap();
        let bytes = b"four".to_vec();
        let metadata = BlobMetadata::from_bytes(&bytes);
        let digest = metadata.digest.clone();
        let mut next = initial.state;
        next.objects
            .insert(digest.clone(), CacheObject::Blob(metadata.clone()));
        let mut transaction = CacheTransaction::metadata_only(0, next);
        transaction.writes.insert(
            digest.clone(),
            ObjectWrite::new(CacheObject::Blob(metadata), Some(bytes)).unwrap(),
        );

        assert!(matches!(
            backend.commit(transaction.clone()).await,
            Err(BackendError::QuotaExceeded { .. })
        ));
        assert_eq!(backend.snapshot().await.unwrap().revision, 0);
        assert_eq!(backend.read_object_bytes(&digest).await.unwrap(), None);

        backend.set_quota(None);
        assert!(matches!(
            backend.commit(transaction.clone()).await.unwrap(),
            CommitOutcome::Committed(_)
        ));
        assert!(matches!(
            backend.commit(transaction).await.unwrap(),
            CommitOutcome::Conflict { actual_revision: 1 }
        ));
        assert_eq!(backend.snapshot().await.unwrap().revision, 1);
    });
}

#[test]
fn metadata_cannot_publish_without_required_payload() {
    block_on(async {
        let backend = MemoryBackend::new();
        let bytes = b"payload";
        let metadata = BlobMetadata::from_bytes(bytes);
        let mut next = backend.snapshot().await.unwrap().state;
        next.objects
            .insert(metadata.digest.clone(), CacheObject::Blob(metadata));
        let error = backend
            .commit(CacheTransaction::metadata_only(0, next))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("lacks an atomic write"));
        assert_eq!(backend.snapshot().await.unwrap().revision, 0);
    });
}
