//! Shared bounded persistence for numerical subtask caches. Domain owners supply
//! opaque identities, accepted bytes and their decoded-storage inventory.
use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
};
use rusqlite::{Connection, ErrorCode};
use std::{
    path::PathBuf,
    sync::{Mutex, TryLockError},
};

mod read;
#[cfg(test)]
mod tests;
mod write;

pub(crate) const CONNECTION_RESIDENT_BYTES: u64 = 2 * 1024 * 1024 + 128 * 1024;
pub(crate) const MAXIMUM_ENTRY_BYTES: u64 = 32 * 1024 * 1024;
const MAXIMUM_PAYLOAD_BYTES: u64 = 1024 * 1024 * 1024;
const MAXIMUM_ENTRIES: u64 = 8192;
const IO_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub(crate) enum CacheError {
    Control(GeometryEvaluationError),
    Storage(String),
}
pub(crate) struct CachedBlob<'a> {
    pub bytes: Vec<u8>,
    pub storage: runmat_geometry_core::GeometryAllocationReservation<'a>,
    pub materialization_resident_bytes: u64,
}

pub(crate) struct BoundedMeshingBlobCache {
    path: PathBuf,
    connection: Mutex<Option<Connection>>,
    maximum_payload_bytes: u64,
    maximum_entries: u64,
}
impl std::fmt::Debug for BoundedMeshingBlobCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundedMeshingBlobCache")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl BoundedMeshingBlobCache {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self {
            path,
            connection: Mutex::new(None),
            maximum_payload_bytes: MAXIMUM_PAYLOAD_BYTES,
            maximum_entries: MAXIMUM_ENTRIES,
        }
    }

    pub(crate) fn admit(
        &self,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<bool, CacheError> {
        admit(control, CONNECTION_RESIDENT_BYTES)
    }

    fn with_connection<T>(
        &self,
        control: &dyn GeometryEvaluationControl,
        action: impl FnOnce(&mut Connection) -> Result<T, CacheError>,
    ) -> Result<Option<T>, CacheError> {
        control.checkpoint().map_err(CacheError::Control)?;
        let mut guard = match self.connection.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => {
                return Err(storage("numerical cache connection lock poisoned"))
            }
        };
        if guard.is_none() {
            match self.open() {
                Ok(connection) => *guard = Some(connection),
                Err(error) if busy(&error) => return Ok(None),
                Err(error) => return Err(storage(error)),
            }
        }
        let result = action(guard.as_mut().expect("connection initialized"));
        control.checkpoint().map_err(CacheError::Control)?;
        result.map(Some)
    }

    fn open(&self) -> Result<Connection, rusqlite::Error> {
        let parent = self
            .path
            .parent()
            .ok_or(rusqlite::Error::InvalidPath(self.path.clone()))?;
        runmat_filesystem::NativeFsProvider
            .create_private_directory_durable(parent)
            .map_err(|_| rusqlite::Error::InvalidPath(parent.to_owned()))?;
        // macOS exposes its temporary directory through /var -> /private/var.
        // Resolve the granted parent; keep NOFOLLOW on the database leaf itself.
        let parent = std::fs::canonicalize(parent)
            .map_err(|_| rusqlite::Error::InvalidPath(parent.to_owned()))?;
        let filename = self
            .path
            .file_name()
            .ok_or_else(|| rusqlite::Error::InvalidPath(self.path.clone()))?;
        let path = parent.join(filename);
        let connection = Connection::open_with_flags(
            &path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX
                | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        connection.busy_timeout(std::time::Duration::ZERO)?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA cache_size=-2048;
             PRAGMA mmap_size=0; PRAGMA wal_autocheckpoint=256; PRAGMA journal_size_limit=2097152;
             PRAGMA max_page_count=294912;
             CREATE TABLE IF NOT EXISTS cache_state (
               singleton INTEGER PRIMARY KEY CHECK(singleton=1),
               payload_bytes INTEGER NOT NULL CHECK(payload_bytes>=0),
               entry_count INTEGER NOT NULL CHECK(entry_count>=0));
             INSERT OR IGNORE INTO cache_state VALUES(1,0,0);
             CREATE TABLE IF NOT EXISTS cache_entries (
               cache_key BLOB UNIQUE NOT NULL CHECK(length(cache_key)=32),
               payload BLOB NOT NULL, digest BLOB NOT NULL CHECK(length(digest)=32),
               payload_bytes INTEGER NOT NULL CHECK(payload_bytes>=0),
               materialization_bytes INTEGER NOT NULL CHECK(materialization_bytes>=0),
               touched INTEGER NOT NULL);
             CREATE INDEX IF NOT EXISTS cache_lru ON cache_entries(touched,cache_key);",
        )?;
        Ok(connection)
    }
}

fn storage(error: impl std::fmt::Display) -> CacheError {
    CacheError::Storage(error.to_string())
}
fn busy(error: &rusqlite::Error) -> bool {
    matches!(error, rusqlite::Error::SqliteFailure(value, _) if matches!(value.code, ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked))
}
fn admit(control: &dyn GeometryEvaluationControl, bytes: u64) -> Result<bool, CacheError> {
    match control.consume_allocation_bytes(bytes) {
        Ok(()) => Ok(true),
        Err(error) if error.kind == GeometryEvaluationErrorKind::AllocationBudgetExceeded => {
            Ok(false)
        }
        Err(error) => Err(CacheError::Control(error)),
    }
}
fn touched() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
        })
}
fn digest(bytes: &[u8], control: &dyn GeometryEvaluationControl) -> Result<[u8; 32], CacheError> {
    use sha2::{Digest as _, Sha256};
    let mut hash = Sha256::new();
    for chunk in bytes.chunks(IO_CHUNK_BYTES) {
        control.checkpoint().map_err(CacheError::Control)?;
        control
            .consume_search_work(1)
            .map_err(CacheError::Control)?;
        hash.update(chunk);
    }
    Ok(hash.finalize().into())
}
