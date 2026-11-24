//! Mesh files belong to the result store. They are written through the captured
//! result transaction when one is open, otherwise through the result store's
//! own provider. The process's installed provider is used only when the result
//! store keeps results in memory and so has no filesystem of its own.
use std::{io::ErrorKind, path::Path};

use super::super::storage::{AnalysisPublicationError, ResultStoreProvider};
use runmat_filesystem::{workspace_access::WorkspaceAccess, AtomicCreateResult};

/// One canonical mesh object and the digest that names it.
pub(super) struct MeshObject<'a> {
    pub(super) path: &'a Path,
    pub(super) bytes: &'a [u8],
    /// The canonical digest of `bytes`, as it appears in the file name.
    pub(super) digest: &'a str,
    /// Decodes stored bytes and recomputes their canonical digest. `None`
    /// means the stored bytes do not decode as this kind of object.
    pub(super) identify: &'a dyn Fn(&[u8]) -> Option<String>,
}

pub(super) struct MeshPersistence {
    scope: Option<Box<dyn WorkspaceAccess>>,
}

impl MeshPersistence {
    #[cfg(test)]
    pub(super) fn capture(root: &Path) -> Result<Self, String> {
        Self::capture_typed(root).map_err(|error| error.to_string())
    }

    pub(super) fn capture_typed(root: &Path) -> Result<Self, AnalysisPublicationError> {
        let provider = ResultStoreProvider::capture()?;
        if matches!(provider, ResultStoreProvider::HostOfMemoryStore(_)) {
            runmat_filesystem::create_dir_all(root).map_err(AnalysisPublicationError::from_io)?;
            return Ok(Self { scope: None });
        }
        let maximum = super::super::configuration::publication_staging_budget()
            .map_err(AnalysisPublicationError::from)?;
        let scope = provider.workspace_access(root, true, maximum)?;
        // A native scope creates its writable root when it opens, and refuses
        // `.` as a target because the root's parent holds the publication
        // journal. A transaction view needs the directory staged.
        if matches!(provider, ResultStoreProvider::Scoped(_)) {
            scope
                .create_dir_all(Path::new("."))
                .map_err(AnalysisPublicationError::from_io)?;
        }
        Ok(Self { scope: Some(scope) })
    }

    pub(super) fn persist_typed(
        &self,
        object: &MeshObject<'_>,
    ) -> Result<(), AnalysisPublicationError> {
        let Some(scope) = &self.scope else {
            return super::persist_canonical(object).map_err(Into::into);
        };
        let relative = Path::new(
            object
                .path
                .file_name()
                .ok_or("mesh artifact has no file name")?,
        );
        let existing_len = match scope.metadata(relative) {
            Ok(metadata) => Some(metadata.len()),
            Err(error) if error.kind() == ErrorKind::NotFound => None,
            Err(error) => return Err(AnalysisPublicationError::from_io(error)),
        };
        if let Some(existing_len) = existing_len {
            if crate::interrupt::is_cancelled() {
                return Err("meshing persistence cancelled".into());
            }
            let existing = if existing_len <= super::stored_object_read_limit(object.bytes) {
                Some(
                    scope
                        .read(relative, existing_len)
                        .map_err(AnalysisPublicationError::from_io)?,
                )
            } else {
                None
            };
            if super::reconcile_existing(object, existing.as_deref())? == super::Reconciled::Current
            {
                return scope
                    .synchronize_immutable(relative)
                    .map_err(AnalysisPublicationError::from_io);
            }
            // Stale or corrupt: replace it. The create below is conditional on
            // the name being free, so a concurrent writer's object is compared,
            // never overwritten.
            scope
                .remove_file(relative)
                .map_err(AnalysisPublicationError::from_io)?;
        }
        if crate::interrupt::is_cancelled() {
            return Err("meshing persistence cancelled".into());
        }
        // Only a concurrent creator needs an owned comparison buffer. Its
        // maximum is the incoming wire length, reserved by the caller's
        // mesh-persistence phase before either encoding is allocated.
        match scope
            .atomic_create_new(relative, object.bytes, object.bytes.len() as u64)
            .map_err(AnalysisPublicationError::from_io)?
        {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing) => {
                if super::reconcile_existing(object, Some(&existing))? != super::Reconciled::Current
                {
                    return Err(format!(
                        "canonical meshing output collision at {}: another writer created a different object while this one was saved",
                        object.path.display()
                    )
                    .into());
                }
            }
        }
        scope
            .synchronize_immutable(relative)
            .map_err(AnalysisPublicationError::from_io)
    }
}
