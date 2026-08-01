use crate::{NativeCacheError, SqliteCacheBackend};
use runmat_package::ContentDigest;
use runmat_package_cache::{acquire_lease, release_lease, renew_lease, Lease, LeaseId, LeaseOwner};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DEFAULT_TTL: Duration = Duration::from_secs(5 * 60);
const TRANSACTION_RETRIES: usize = 16;
static NEXT_LEASE: AtomicU64 = AtomicU64::new(1);

/// Keeps a cache object closure live for the lifetime of a native composition.
///
/// The renewable metadata lease protects both transactional payloads and any
/// content-addressed native tree materialization rooted at those objects. Drop
/// stops renewal and releases the lease synchronously; process death is handled
/// by TTL expiry.
pub struct NativeCacheLease {
    backend: Arc<SqliteCacheBackend>,
    lease: Arc<Mutex<Lease>>,
    stop: Option<mpsc::Sender<()>>,
    worker: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for NativeCacheLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeCacheLease")
            .field("lease", &self.lease.lock().ok().map(|lease| lease.clone()))
            .finish_non_exhaustive()
    }
}

impl NativeCacheLease {
    pub async fn acquire(
        backend: Arc<SqliteCacheBackend>,
        objects: BTreeSet<ContentDigest>,
    ) -> Result<Option<Self>, NativeCacheError> {
        Self::acquire_with_ttl(backend, objects, DEFAULT_TTL).await
    }

    pub async fn acquire_with_ttl(
        backend: Arc<SqliteCacheBackend>,
        objects: BTreeSet<ContentDigest>,
        ttl: Duration,
    ) -> Result<Option<Self>, NativeCacheError> {
        if objects.is_empty() {
            return Ok(None);
        }
        let sequence = NEXT_LEASE.fetch_add(1, Ordering::Relaxed);
        let identity = format!("native-{}-{sequence}-{}", std::process::id(), now_ms());
        let lease = acquire_lease(
            &backend,
            LeaseId::new(identity.clone()).expect("generated lease id is valid"),
            LeaseOwner::new(format!("process-{}", std::process::id()))
                .expect("generated lease owner is valid"),
            objects,
            now_ms(),
            duration_ms(ttl),
            TRANSACTION_RETRIES,
        )
        .await?;
        let lease = Arc::new(Mutex::new(lease));
        let (stop, receiver) = mpsc::channel();
        let worker_backend = backend.clone();
        let worker_lease = lease.clone();
        let interval = std::cmp::max(ttl / 3, Duration::from_millis(10));
        let worker = std::thread::spawn(move || loop {
            match receiver.recv_timeout(interval) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            let current = match worker_lease.lock() {
                Ok(lease) => lease.clone(),
                Err(_) => break,
            };
            let renewed = futures::executor::block_on(renew_lease(
                &worker_backend,
                &current,
                now_ms(),
                duration_ms(ttl),
                TRANSACTION_RETRIES,
            ));
            if let Ok(renewed) = renewed {
                if let Ok(mut lease) = worker_lease.lock() {
                    *lease = renewed;
                }
            }
        });
        Ok(Some(Self {
            backend,
            lease,
            stop: Some(stop),
            worker: Some(worker),
        }))
    }

    pub fn lease(&self) -> Option<Lease> {
        self.lease.lock().ok().map(|lease| lease.clone())
    }
}

impl Drop for NativeCacheLease {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        if let Ok(lease) = self.lease.lock().map(|lease| lease.clone()) {
            let backend = self.backend.clone();
            let _ = std::thread::spawn(move || {
                futures::executor::block_on(release_lease(&backend, &lease, TRANSACTION_RETRIES))
            })
            .join();
        }
    }
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
