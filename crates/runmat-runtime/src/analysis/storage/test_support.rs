//! Explicit artifact ownership for tests that do not exercise filesystem persistence.
use super::*;

pub(crate) struct IsolatedArtifactStore {
    previous: Option<Arc<dyn AnalysisArtifactStore>>,
    previous_retention: Option<AnalysisArtifactRetentionConfig>,
    _lock: MutexGuard<'static, ()>,
    // Released after the store lock (field drop order).
    _provider_lock: MutexGuard<'static, ()>,
}

/// Hold the shared host locks, use a fresh store, and restore the previous host
/// configuration on exit. No operator artifacts are read, changed or removed.
///
/// The filesystem provider lock is taken first. Tests that install their own
/// provider do so under this guard instead of locking the provider again, and
/// tests that read real directories see the provider the process started with.
pub(crate) fn isolated_artifact_store() -> IsolatedArtifactStore {
    let provider_lock = runmat_filesystem::provider_override_lock();
    let lock = artifact_store_test_guard();
    let previous = std::mem::replace(
        &mut *global_store().write().unwrap_or_else(|e| e.into_inner()),
        Arc::new(InMemoryAnalysisArtifactStore::new()),
    );
    let previous_retention = std::mem::take(
        &mut *retention_config()
            .write()
            .unwrap_or_else(|e| e.into_inner()),
    );
    IsolatedArtifactStore {
        previous: Some(previous),
        previous_retention: Some(previous_retention),
        _lock: lock,
        _provider_lock: provider_lock,
    }
}

impl Drop for IsolatedArtifactStore {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            *global_store().write().unwrap_or_else(|e| e.into_inner()) = previous;
        }
        if let Some(previous) = self.previous_retention.take() {
            *retention_config()
                .write()
                .unwrap_or_else(|e| e.into_inner()) = previous;
        }
    }
}
