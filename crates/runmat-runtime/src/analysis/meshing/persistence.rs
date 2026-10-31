//! Mesh files share the result transaction when publication has captured one.
use std::{io::ErrorKind, path::Path};

use runmat_filesystem::{workspace_access::WorkspaceAccess, AtomicCreateResult};

pub(super) struct MeshPersistence {
    scope: Option<Box<dyn WorkspaceAccess>>,
}

impl MeshPersistence {
    pub(super) fn capture(root: &Path) -> Result<Self, String> {
        let scope = super::super::storage::scoped_artifact_provider()
            .map(|provider| {
                let maximum = super::super::configuration::publication_staging_budget()
                    .map_err(|error| error.to_string())?;
                provider
                    .workspace_access(root, true, maximum)
                    .map_err(|error| format!("failed to capture {}: {error}", root.display()))
            })
            .transpose()?;
        match &scope {
            Some(scope) => scope.create_dir_all(Path::new(".")),
            None => runmat_filesystem::create_dir_all(root),
        }
        .map_err(|error| format!("failed to create {}: {error}", root.display()))?;
        Ok(Self { scope })
    }

    pub(super) fn persist(&self, path: &Path, bytes: &[u8]) -> Result<(), String> {
        let Some(scope) = &self.scope else {
            return super::persist_canonical(path, bytes);
        };
        let relative = Path::new(path.file_name().ok_or("mesh artifact has no file name")?);
        let exists = match scope.metadata(relative) {
            Ok(_) => true,
            Err(error) if error.kind() == ErrorKind::NotFound => false,
            Err(error) => return Err(format!("failed to inspect {}: {error}", path.display())),
        };
        if exists {
            let mut existing = scope
                .open_read(relative)
                .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
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
                .map_err(|error| format!("failed to persist {}: {error}", path.display()))?
            {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) => {
                    super::compare_canonical(path, bytes, &mut std::io::Cursor::new(existing))?;
                }
            }
        }
        scope
            .synchronize_immutable(relative)
            .map_err(|error| format!("failed to confirm {}: {error}", path.display()))
    }
}
