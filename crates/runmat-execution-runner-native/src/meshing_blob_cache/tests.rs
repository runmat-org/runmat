use super::*;
mod diagnostic_export;
mod session_retention;
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "runmat-meshing-blob-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
    fn cache(&self) -> BoundedMeshingBlobCache {
        BoundedMeshingBlobCache::new(self.0.join("cache.sqlite"))
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[derive(Default)]
struct Control {
    limit: Option<u64>,
    live: std::sync::atomic::AtomicU64,
    checkpoints: AtomicUsize,
    cancel_at: Option<usize>,
}
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        let count = self.checkpoints.fetch_add(1, Ordering::Relaxed);
        if self.cancel_at.is_some_and(|limit| count >= limit) {
            Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::Cancelled,
                "cache test cancellation",
            ))
        } else {
            Ok(())
        }
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()?;
        self.live
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current
                    .checked_add(bytes)
                    .filter(|next| self.limit.is_none_or(|limit| *next <= limit))
            })
            .map(|_| ())
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                    "cache test ceiling",
                )
            })
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        assert!(self.live.fetch_sub(bytes, Ordering::Relaxed) >= bytes);
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }
}
// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A reopened native store preserves exact retained bytes and decoded-storage inventory,
//   and rejects conflicting bytes under one key. It does not independently establish numerical
//   provenance or acceptance.
#[test]
fn reopened_store_preserves_original_bytes_and_rejects_identity_collision() {
    let fixture = Fixture::new();
    let control = Control::default();
    let key = [7; 32];
    {
        let cache = fixture.cache();
        assert!(cache.admit(&control).unwrap());
        assert!(cache
            .write(&key, b"original accepted bytes", 8192, &control)
            .unwrap());
    }
    let cache = fixture.cache();
    let entry = cache.read(&key, &control).unwrap().unwrap();
    assert_eq!(entry.bytes, b"original accepted bytes");
    assert_eq!(entry.materialization_resident_bytes, 8192);
    assert!(cache
        .write(&key, b"original accepted bytes", 8192, &control)
        .unwrap());
    assert!(matches!(
        cache.write(&key, b"other bytes", 8192, &control),
        Err(CacheError::Storage(_))
    ));
    assert!(matches!(
        cache.write(&key, b"original accepted bytes", 8193, &control),
        Err(CacheError::Storage(_))
    ));
}
#[test]
fn denied_materialization_skips_read_before_allocation_and_cancellation_propagates() {
    let fixture = Fixture::new();
    let cache = fixture.cache();
    let key = [8; 32];
    let control = Control::default();
    cache.write(&key, b"bytes", 5000, &control).unwrap();
    assert!(cache
        .read(
            &key,
            &Control {
                limit: Some(5000),
                ..Control::default()
            }
        )
        .unwrap()
        .is_none());
    assert!(matches!(
        cache.read(
            &key,
            &Control {
                cancel_at: Some(0),
                ..Control::default()
            }
        ),
        Err(CacheError::Control(_))
    ));
    assert_eq!(cache.read(&key, &control).unwrap().unwrap().bytes, b"bytes");
}
#[test]
fn quota_eviction_is_bounded_and_transactional() {
    let fixture = Fixture::new();
    let mut cache = fixture.cache();
    cache.maximum_payload_bytes = 8;
    cache.maximum_entries = 2;
    let control = Control::default();
    for key in [1, 2, 3] {
        assert!(cache.write(&[key; 32], b"1234", 50, &control).unwrap());
    }
    let entries = (1..=3)
        .filter(|key| cache.read(&[*key; 32], &control).unwrap().is_some())
        .count();
    assert_eq!(entries, 2);
    assert!(!cache.write(&[4; 32], b"too large", 50, &control).unwrap());
    let state = cache
        .with_connection(&control, |connection| {
            connection
                .query_row(
                    "SELECT payload_bytes,entry_count FROM cache_state",
                    [],
                    |row| Ok((row.get::<_, u64>(0)?, row.get::<_, u64>(1)?)),
                )
                .map_err(storage)
        })
        .unwrap()
        .unwrap();
    assert_eq!(state, (8, 2));
}
// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Cancellation during incremental storage rolls back partial payload and quota updates
//   while preserving previously retained bytes. This bounds one storage transaction, not all native
//   task cancellation latency.
#[test]
fn cancellation_rolls_back_partial_blob_and_quota_changes() {
    let fixture = Fixture::new();
    let cache = fixture.cache();
    let control = Control::default();
    cache.write(&[1; 32], b"retained", 50, &control).unwrap();
    let bytes = vec![42; IO_CHUNK_BYTES * 3];
    let cancelled = Control {
        cancel_at: Some(10),
        ..Control::default()
    };
    assert!(matches!(
        cache.write(&[2; 32], &bytes, 500, &cancelled),
        Err(CacheError::Control(_))
    ));
    assert!(cache.read(&[2; 32], &control).unwrap().is_none());
    assert_eq!(
        cache.read(&[1; 32], &control).unwrap().unwrap().bytes,
        b"retained"
    );
}
#[test]
fn corrupt_payload_is_rejected_and_busy_writer_does_not_block_numerics() {
    let fixture = Fixture::new();
    let cache = fixture.cache();
    let control = Control::default();
    let key = [4; 32];
    cache.write(&key, b"valid", 50, &control).unwrap();
    let connection = Connection::open(&cache.path).unwrap();
    connection
        .execute("UPDATE cache_entries SET payload=?1", [b"wrong".as_slice()])
        .unwrap();
    assert!(matches!(
        cache.read(&key, &control),
        Err(CacheError::Storage(_))
    ));
    assert!(cache.write(&key, b"valid", 50, &control).unwrap());
    assert_eq!(cache.read(&key, &control).unwrap().unwrap().bytes, b"valid");
    connection.execute_batch("BEGIN IMMEDIATE").unwrap();
    let other = fixture.cache();
    assert!(!other.write(&[5; 32], b"new", 50, &control).unwrap());
    connection.execute_batch("ROLLBACK").unwrap();
    assert!(other.write(&[5; 32], b"new", 50, &control).unwrap());
}

#[cfg(unix)]
#[test]
fn ancestor_alias_resolves_while_private_directory_and_database_leaves_reject_symlinks() {
    let fixture = Fixture::new();
    let real = fixture.0.join("real");
    std::fs::create_dir_all(real.join("authority")).unwrap();
    let alias = fixture.0.join("alias");
    std::os::unix::fs::symlink(&real, &alias).unwrap();
    let control = Control::default();
    let cache = BoundedMeshingBlobCache::new(alias.join("authority/cache.sqlite"));
    assert!(cache.write(&[3; 32], b"accepted", 50, &control).unwrap());
    assert_eq!(
        cache.read(&[3; 32], &control).unwrap().unwrap().bytes,
        b"accepted"
    );
    drop(cache);
    std::fs::rename(
        real.join("authority/cache.sqlite"),
        real.join("authority/original.sqlite"),
    )
    .unwrap();
    std::os::unix::fs::symlink(
        real.join("authority/original.sqlite"),
        real.join("authority/cache.sqlite"),
    )
    .unwrap();
    let private_directory_alias = BoundedMeshingBlobCache::new(alias.join("bad.sqlite"));
    assert!(matches!(
        private_directory_alias.read(&[3; 32], &control),
        Err(CacheError::Storage(_))
    ));
    let forbidden = BoundedMeshingBlobCache::new(alias.join("authority/cache.sqlite"));
    assert!(matches!(
        forbidden.read(&[3; 32], &control),
        Err(CacheError::Storage(_))
    ));
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Repeated native cache reads release encoded buffers and unused decode workspace. A
//   simultaneously live entry or conservatively retained decoded result cannot escape admission.
//   This is requested storage accounting, not measured RSS.
#[test]
fn repeated_reads_release_encoded_storage_but_retained_materialization_stays_charged() {
    let fixture = Fixture::new();
    let cache = fixture.cache();
    let control = Control {
        limit: Some(100),
        ..Control::default()
    };
    cache.write(&[9; 32], b"0123456789", 90, &control).unwrap();
    for _ in 0..100 {
        let entry = cache.read(&[9; 32], &control).unwrap().unwrap();
        assert_eq!(control.live.load(Ordering::Relaxed), 100);
        assert!(cache.read(&[9; 32], &control).unwrap().is_none());
        drop(entry);
        assert_eq!(control.live.load(Ordering::Relaxed), 0);
    }
    let mut entry = cache.read(&[9; 32], &control).unwrap().unwrap();
    entry
        .storage
        .retain_bytes(entry.materialization_resident_bytes);
    drop(entry);
    assert_eq!(control.live.load(Ordering::Relaxed), 90);
    assert!(cache.read(&[9; 32], &control).unwrap().is_none());
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Corrupt and cancelled reads release unpublished storage while preserving typed
//   errors. No invalid cache payload becomes numerical authority.
#[test]
fn corrupt_reads_and_cancelled_reads_release_the_whole_unpublished_inventory() {
    let fixture = Fixture::new();
    let cache = fixture.cache();
    let control = Control {
        limit: Some(100),
        ..Control::default()
    };
    cache.write(&[9; 32], b"0123456789", 90, &control).unwrap();
    let connection = Connection::open(&cache.path).unwrap();
    connection
        .execute("UPDATE cache_entries SET digest=zeroblob(32)", [])
        .unwrap();
    assert!(matches!(
        cache.read(&[9; 32], &control),
        Err(CacheError::Storage(_))
    ));
    assert_eq!(control.live.load(Ordering::Relaxed), 0);
    // with_connection, reservation and allocation checkpoints precede the first
    // blob-read checkpoint. Cancellation there must unwind its reservation.
    let cancelled = Control {
        limit: Some(100),
        cancel_at: Some(3),
        ..Control::default()
    };
    assert!(matches!(
        cache.read(&[9; 32], &cancelled),
        Err(CacheError::Control(_))
    ));
    assert_eq!(cancelled.live.load(Ordering::Relaxed), 0);
}
