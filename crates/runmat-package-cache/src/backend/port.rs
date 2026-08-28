use super::{BackendSnapshot, CacheTransaction, CommitOutcome};
use crate::BackendError;
use futures::future::LocalBoxFuture;
use runmat_package::ContentDigest;

pub trait CacheBackend: Send + Sync {
    fn snapshot(&self) -> LocalBoxFuture<'_, Result<BackendSnapshot, BackendError>>;

    fn commit(
        &self,
        transaction: CacheTransaction,
    ) -> LocalBoxFuture<'_, Result<CommitOutcome, BackendError>>;

    fn read_object_bytes(
        &self,
        digest: &ContentDigest,
    ) -> LocalBoxFuture<'_, Result<Option<Vec<u8>>, BackendError>>;
}

impl<T> CacheBackend for std::sync::Arc<T>
where
    T: CacheBackend + ?Sized,
{
    fn snapshot(&self) -> LocalBoxFuture<'_, Result<BackendSnapshot, BackendError>> {
        (**self).snapshot()
    }

    fn commit(
        &self,
        transaction: CacheTransaction,
    ) -> LocalBoxFuture<'_, Result<CommitOutcome, BackendError>> {
        (**self).commit(transaction)
    }

    fn read_object_bytes(
        &self,
        digest: &ContentDigest,
    ) -> LocalBoxFuture<'_, Result<Option<Vec<u8>>, BackendError>> {
        (**self).read_object_bytes(digest)
    }
}
