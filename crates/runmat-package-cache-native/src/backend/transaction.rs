use super::{objects, recovery};
use crate::SqliteCacheBackend;
use futures::future::{ready, LocalBoxFuture};
use runmat_package::ContentDigest;
use runmat_package_cache::{
    BackendCommit, BackendError, BackendSnapshot, CacheBackend, CacheState, CacheTransaction,
    CommitOutcome,
};
use rusqlite::{OptionalExtension, TransactionBehavior};

impl CacheBackend for SqliteCacheBackend {
    fn snapshot(&self) -> LocalBoxFuture<'_, Result<BackendSnapshot, BackendError>> {
        Box::pin(ready(snapshot(self)))
    }

    fn commit(
        &self,
        transaction: CacheTransaction,
    ) -> LocalBoxFuture<'_, Result<CommitOutcome, BackendError>> {
        Box::pin(ready(commit(self, transaction)))
    }

    fn read_object_bytes(
        &self,
        digest: &ContentDigest,
    ) -> LocalBoxFuture<'_, Result<Option<Vec<u8>>, BackendError>> {
        Box::pin(ready(read(self, digest)))
    }
}

fn snapshot(backend: &SqliteCacheBackend) -> Result<BackendSnapshot, BackendError> {
    let connection = backend.connection.lock().map_err(lock_error)?;
    let (revision, state) = read_state(&connection)?;
    recovery::validate_payload_closure(&connection, &state).map_err(cache_error)?;
    Ok(BackendSnapshot { revision, state })
}

fn commit(
    backend: &SqliteCacheBackend,
    transaction: CacheTransaction,
) -> Result<CommitOutcome, BackendError> {
    let mut connection = backend.connection.lock().map_err(lock_error)?;
    let sql = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(database_error)?;
    let (revision, current) = read_state(&sql)?;
    if revision != transaction.expected_revision {
        return Ok(CommitOutcome::Conflict {
            actual_revision: revision,
        });
    }
    transaction
        .validate_transition(&current)
        .map_err(cache_error)?;
    for digest in &transaction.deletes {
        sql.execute(
            "DELETE FROM object_payloads WHERE digest = ?1",
            [digest.to_string()],
        )
        .map_err(database_error)?;
    }
    for (digest, write) in &transaction.writes {
        if let Some(bytes) = &write.bytes {
            sql.execute(
                "INSERT INTO object_payloads (digest, bytes) VALUES (?1, ?2)
                 ON CONFLICT(digest) DO UPDATE SET bytes = excluded.bytes",
                rusqlite::params![digest.to_string(), bytes],
            )
            .map_err(database_error)?;
        } else {
            sql.execute(
                "DELETE FROM object_payloads WHERE digest = ?1",
                [digest.to_string()],
            )
            .map_err(database_error)?;
        }
    }
    let used = objects::stored_bytes(&sql).map_err(database_error)?;
    if let Some(quota) = backend.quota_bytes {
        if used > quota {
            return Err(BackendError::QuotaExceeded {
                requested_bytes: used - quota,
                available_bytes: quota.saturating_sub(current.total_stored_payload_bytes()),
            });
        }
    }
    let next_revision = revision
        .checked_add(1)
        .ok_or_else(|| BackendError::Failure("cache revision overflow".to_string()))?;
    let state_json = serde_json::to_vec(&transaction.next_state)
        .map_err(|error| BackendError::Failure(error.to_string()))?;
    sql.execute(
        "UPDATE cache_state SET revision = ?1, state_json = ?2 WHERE singleton = 1",
        rusqlite::params![next_revision, state_json],
    )
    .map_err(database_error)?;
    sql.commit().map_err(database_error)?;
    Ok(CommitOutcome::Committed(BackendCommit {
        revision: next_revision,
    }))
}

fn read(
    backend: &SqliteCacheBackend,
    digest: &ContentDigest,
) -> Result<Option<Vec<u8>>, BackendError> {
    let mut connection = backend.connection.lock().map_err(lock_error)?;
    let transaction = connection.transaction().map_err(database_error)?;
    objects::read(&transaction, digest).map_err(database_error)
}

fn read_state(connection: &rusqlite::Connection) -> Result<(u64, CacheState), BackendError> {
    let row: Option<(i64, Vec<u8>)> = connection
        .query_row(
            "SELECT revision, state_json FROM cache_state WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let (revision, bytes) =
        row.ok_or_else(|| BackendError::Failure("cache state row is missing".to_string()))?;
    let revision = u64::try_from(revision)
        .map_err(|_| BackendError::Failure("cache revision is negative".to_string()))?;
    let state: CacheState = serde_json::from_slice(&bytes)
        .map_err(|error| BackendError::IncompatibleSchema(error.to_string()))?;
    state
        .validate()
        .map_err(|error| BackendError::IncompatibleSchema(error.to_string()))?;
    Ok((revision, state))
}

fn database_error(error: rusqlite::Error) -> BackendError {
    BackendError::Failure(error.to_string())
}

fn cache_error(error: runmat_package_cache::CacheError) -> BackendError {
    BackendError::Failure(error.to_string())
}

fn lock_error<T>(error: std::sync::PoisonError<T>) -> BackendError {
    BackendError::Failure(format!("cache database lock poisoned: {error}"))
}
