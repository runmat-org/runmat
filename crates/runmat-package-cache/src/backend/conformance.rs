//! Reusable backend contract tests and an in-memory reference backend.

use super::{
    BackendCommit, BackendSnapshot, CacheBackend, CacheTransaction, CommitOutcome, ObjectWrite,
};
use crate::object::CacheObject;
use crate::{BackendError, CacheError};
use futures::future::{ready, LocalBoxFuture};
use runmat_package::ContentDigest;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct MemoryBackend {
    inner: Arc<Mutex<MemoryState>>,
}

#[derive(Debug)]
struct MemoryState {
    revision: u64,
    state: crate::CacheState,
    bytes: BTreeMap<ContentDigest, Vec<u8>>,
    quota_bytes: Option<u64>,
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MemoryState {
                revision: 0,
                state: crate::CacheState::default(),
                bytes: BTreeMap::new(),
                quota_bytes: None,
            })),
        }
    }

    pub fn with_quota(quota_bytes: u64) -> Self {
        let backend = Self::new();
        backend.set_quota(Some(quota_bytes));
        backend
    }

    pub fn set_quota(&self, quota_bytes: Option<u64>) {
        self.inner.lock().expect("memory backend lock").quota_bytes = quota_bytes;
    }

    /// Simulates an out-of-band browser eviction or damaged native object file.
    pub fn evict_payload(&self, digest: &ContentDigest) {
        self.inner
            .lock()
            .expect("memory backend lock")
            .bytes
            .remove(digest);
    }
}

impl CacheBackend for MemoryBackend {
    fn snapshot(&self) -> LocalBoxFuture<'_, Result<BackendSnapshot, BackendError>> {
        let result = self
            .inner
            .lock()
            .map_err(lock_error)
            .map(|inner| BackendSnapshot {
                revision: inner.revision,
                state: inner.state.clone(),
            });
        Box::pin(ready(result))
    }

    fn commit(
        &self,
        transaction: CacheTransaction,
    ) -> LocalBoxFuture<'_, Result<CommitOutcome, BackendError>> {
        let result = commit_memory(&self.inner, transaction);
        Box::pin(ready(result))
    }

    fn read_object_bytes(
        &self,
        digest: &ContentDigest,
    ) -> LocalBoxFuture<'_, Result<Option<Vec<u8>>, BackendError>> {
        let result = self
            .inner
            .lock()
            .map_err(lock_error)
            .map(|inner| inner.bytes.get(digest).cloned());
        Box::pin(ready(result))
    }
}

fn commit_memory(
    shared: &Mutex<MemoryState>,
    transaction: CacheTransaction,
) -> Result<CommitOutcome, BackendError> {
    let mut inner = shared.lock().map_err(lock_error)?;
    if transaction.expected_revision != inner.revision {
        return Ok(CommitOutcome::Conflict {
            actual_revision: inner.revision,
        });
    }
    transaction
        .validate_transition(&inner.state)
        .map_err(cache_error)?;

    let mut next_bytes = inner.bytes.clone();
    for digest in &transaction.deletes {
        next_bytes.remove(digest);
    }
    for (digest, write) in &transaction.writes {
        match &write.bytes {
            Some(bytes) => {
                next_bytes.insert(digest.clone(), bytes.clone());
            }
            None => {
                next_bytes.remove(digest);
            }
        }
    }
    let used = next_bytes.values().try_fold(0u64, |total, bytes| {
        total
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| BackendError::Failure("cache byte count overflow".to_string()))
    })?;
    if let Some(quota) = inner.quota_bytes {
        if used > quota {
            return Err(BackendError::QuotaExceeded {
                requested_bytes: used - quota,
                available_bytes: quota
                    .saturating_sub(inner.bytes.values().map(|bytes| bytes.len() as u64).sum()),
            });
        }
    }

    inner.state = transaction.next_state;
    inner.bytes = next_bytes;
    inner.revision = inner
        .revision
        .checked_add(1)
        .ok_or_else(|| BackendError::Failure("cache revision overflow".to_string()))?;
    Ok(CommitOutcome::Committed(BackendCommit {
        revision: inner.revision,
    }))
}

fn lock_error<T>(error: std::sync::PoisonError<T>) -> BackendError {
    BackendError::Failure(format!("memory backend lock poisoned: {error}"))
}

fn cache_error(error: CacheError) -> BackendError {
    BackendError::Failure(error.to_string())
}

/// Backend-neutral checks adapters can invoke from their own integration tests.
pub async fn verify_backend_contract<B, F>(make_backend: F) -> Result<(), String>
where
    B: CacheBackend,
    F: Fn() -> B,
{
    let backend = make_backend();
    let initial = backend
        .snapshot()
        .await
        .map_err(|error| error.to_string())?;
    if initial.revision != 0 || initial.state != crate::CacheState::default() {
        return Err("new backend did not start at revision zero with empty state".to_string());
    }

    let bytes = b"backend-conformance".to_vec();
    let metadata = crate::BlobMetadata::from_bytes(&bytes);
    let digest = metadata.digest.clone();
    let mut state = initial.state;
    state
        .objects
        .insert(digest.clone(), CacheObject::Blob(metadata.clone()));
    let mut transaction = CacheTransaction::metadata_only(0, state);
    transaction.writes.insert(
        digest.clone(),
        ObjectWrite::new(CacheObject::Blob(metadata), Some(bytes.clone()))
            .map_err(|error| error.to_string())?,
    );
    match backend
        .commit(transaction.clone())
        .await
        .map_err(|error| error.to_string())?
    {
        CommitOutcome::Committed(commit) if commit.revision == 1 => {}
        outcome => return Err(format!("first commit returned {outcome:?}")),
    }
    if backend
        .read_object_bytes(&digest)
        .await
        .map_err(|error| error.to_string())?
        != Some(bytes)
    {
        return Err("committed payload was not readable".to_string());
    }
    match backend
        .commit(transaction)
        .await
        .map_err(|error| error.to_string())?
    {
        CommitOutcome::Conflict { actual_revision: 1 } => {}
        outcome => return Err(format!("stale commit returned {outcome:?}")),
    }
    let final_snapshot = backend
        .snapshot()
        .await
        .map_err(|error| error.to_string())?;
    if final_snapshot.revision != 1 || !final_snapshot.state.objects.contains_key(&digest) {
        return Err("conflict changed committed backend state".to_string());
    }

    let mut invalid_state = final_snapshot.state.clone();
    let orphan_bytes = b"orphan".to_vec();
    let orphan = crate::BlobMetadata::from_bytes(&orphan_bytes);
    invalid_state
        .objects
        .insert(orphan.digest.clone(), CacheObject::Blob(orphan));
    if backend
        .commit(CacheTransaction::metadata_only(1, invalid_state))
        .await
        .is_ok()
    {
        return Err(
            "backend published payload metadata without an atomic payload write".to_string(),
        );
    }
    if backend
        .snapshot()
        .await
        .map_err(|error| error.to_string())?
        .revision
        != 1
    {
        return Err("rejected transaction changed backend revision".to_string());
    }

    let mut empty = final_snapshot.state;
    empty.objects.remove(&digest);
    let mut delete = CacheTransaction::metadata_only(1, empty);
    delete.deletes.insert(digest.clone());
    match backend
        .commit(delete)
        .await
        .map_err(|error| error.to_string())?
    {
        CommitOutcome::Committed(commit) if commit.revision == 2 => {}
        outcome => return Err(format!("delete commit returned {outcome:?}")),
    }
    if backend
        .read_object_bytes(&digest)
        .await
        .map_err(|error| error.to_string())?
        .is_some()
    {
        return Err("deleted payload remained readable".to_string());
    }
    Ok(())
}
