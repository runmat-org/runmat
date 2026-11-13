//! Mesh files share the result transaction when publication has captured one.
use std::{io::ErrorKind, path::Path};

use super::super::storage::AnalysisPublicationError;
use runmat_filesystem::{workspace_access::WorkspaceAccess, AtomicCreateResult};

pub(super) struct MeshPersistence {
    scope: Option<Box<dyn WorkspaceAccess>>,
}

impl MeshPersistence {
    #[cfg(test)]
    pub(super) fn capture(root: &Path) -> Result<Self, String> {
        Self::capture_typed(root).map_err(|error| error.to_string())
    }

    pub(super) fn capture_typed(root: &Path) -> Result<Self, AnalysisPublicationError> {
        let scope = super::super::storage::scoped_artifact_provider()
            .map(|provider| {
                let maximum = super::super::configuration::publication_staging_budget()
                    .map_err(AnalysisPublicationError::from)?;
                provider
                    .workspace_access(root, true, maximum)
                    .map_err(AnalysisPublicationError::from_io)
            })
            .transpose()?;
        match &scope {
            Some(scope) => scope.create_dir_all(Path::new(".")),
            None => runmat_filesystem::create_dir_all(root),
        }
        .map_err(AnalysisPublicationError::from_io)?;
        Ok(Self { scope })
    }

    #[cfg(test)]
    pub(super) fn persist(&self, path: &Path, bytes: &[u8]) -> Result<(), String> {
        self.persist_typed(path, bytes)
            .map_err(|error| error.to_string())
    }

    pub(super) fn persist_typed(
        &self,
        path: &Path,
        bytes: &[u8],
    ) -> Result<(), AnalysisPublicationError> {
        let Some(scope) = &self.scope else {
            return super::persist_canonical(path, bytes).map_err(Into::into);
        };
        let relative = Path::new(path.file_name().ok_or("mesh artifact has no file name")?);
        let exists = match scope.metadata(relative) {
            Ok(_) => true,
            Err(error) if error.kind() == ErrorKind::NotFound => false,
            Err(error) => return Err(AnalysisPublicationError::from_io(error)),
        };
        if exists {
            let mut existing = scope
                .open_read(relative)
                .map_err(AnalysisPublicationError::from_io)?;
            super::compare_canonical(path, bytes, &mut existing)?;
        } else {
            if crate::interrupt::is_cancelled() {
                return Err("meshing persistence cancelled".into());
            }
            // Only a concurrent creator needs an owned comparison buffer. Its
            // maximum is the incoming wire length, reserved by the caller's
            // mesh-persistence phase before either encoding is allocated.
            match scope
                .atomic_create_new(relative, bytes, bytes.len() as u64)
                .map_err(AnalysisPublicationError::from_io)?
            {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) => {
                    super::compare_canonical(path, bytes, &mut std::io::Cursor::new(existing))?;
                }
            }
        }
        scope
            .synchronize_immutable(relative)
            .map_err(AnalysisPublicationError::from_io)
    }
}
