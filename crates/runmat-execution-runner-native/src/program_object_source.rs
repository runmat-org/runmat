//! Read-only result sources retain session storage, not the driver or worker pool.
use std::{path::PathBuf, sync::Arc};

use runmat_execution::Digest;
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult};

use crate::FilesystemObjectStore;

pub(crate) struct SessionStoreLifetime {
    root: PathBuf,
    retained_for_recovery: std::sync::atomic::AtomicBool,
}

impl SessionStoreLifetime {
    /// The driver supplies its freshly created, validated per-session directory.
    pub(crate) fn new(root: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            root,
            retained_for_recovery: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// A private scratch whose process tree could not be reaped must remain
    /// recoverable; never recursively erase it as ordinary session data.
    pub(crate) fn retain_for_recovery(&self) {
        self.retained_for_recovery
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

impl Drop for SessionStoreLifetime {
    fn drop(&mut self) {
        if !self
            .retained_for_recovery
            .load(std::sync::atomic::Ordering::Acquire)
        {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}

/// A verified object reader that stays valid after its producing session finishes.
/// The last session/source owner releases the temporary session directory. Persist
/// the declared object closure separately when results must survive process exit.
#[derive(Clone)]
pub struct NativeProgramObjectSource {
    store: FilesystemObjectStore,
    _lifetime: Arc<SessionStoreLifetime>,
}

impl NativeProgramObjectSource {
    pub(crate) fn new(store: FilesystemObjectStore, lifetime: Arc<SessionStoreLifetime>) -> Self {
        Self {
            store,
            _lifetime: lifetime,
        }
    }
}

impl CacheImport for NativeProgramObjectSource {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified(digest)
    }
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.store
            .read_verified_bounded_with_admission(digest, maximum_bytes, admit)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.store.read_verified_bounded(digest, maximum_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NativeExecutionConfig, NativeProgramSession};
    use runmat_execution_artifact::{cache::CacheExport, LogicalObject, ObjectNamespace};

    #[test]
    fn result_source_outlives_session_and_cleans_up_after_last_reader() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.store_root = directory.path().join("sessions");
        let session = NativeProgramSession::new(config).unwrap();
        let mut store = session.object_store();
        let session_root = store.root().parent().unwrap().to_path_buf();
        let object = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "retained-result",
            "application/octet-stream",
            b"completed numerical result".to_vec(),
        )
        .unwrap();
        store.write_verified(&object).unwrap();
        let source = session.object_source();
        let another_reader = source.clone();
        drop(session);
        assert!(session_root.is_dir());
        assert!(matches!(
            source.read_verified_bounded(object.descriptor.digest, 0),
            Err(runmat_execution_artifact::ArtifactError::Limit(_))
        ));
        assert_eq!(
            source
                .read_verified_bounded(object.descriptor.digest, object.bytes.len() as u64)
                .unwrap(),
            Some(object.bytes.clone())
        );
        assert_eq!(
            source.read_verified(object.descriptor.digest).unwrap(),
            Some(object.bytes.clone())
        );
        drop(source);
        assert_eq!(
            another_reader
                .read_verified(object.descriptor.digest)
                .unwrap(),
            Some(object.bytes)
        );
        drop(another_reader);
        assert!(!session_root.exists());
        assert!(directory.path().is_dir());
    }
}
