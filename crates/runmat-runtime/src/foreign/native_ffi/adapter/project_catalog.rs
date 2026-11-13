//! Immutable, metadata-only replay authority for one published native project.

use super::*;
use std::path::{Path, PathBuf};

use super::super::NativeProjectLoadRequest;

#[derive(Clone, Default)]
pub(super) struct NativeProjectCatalog {
    pub(super) loads: Vec<NativeProjectLoadRequest>,
    pub(super) lease: Option<runmat_process_host::private_bytes::PrivateByteLease>,
}

impl NativeProjectCatalog {
    pub(super) fn bind_private_root(
        &mut self,
        lease: runmat_process_host::private_bytes::PrivateByteLease,
    ) -> Result<(), RuntimeError> {
        let root = std::fs::canonicalize(lease.path()).map_err(|error| {
            invalid_call(format!("native private root is unavailable: {error}"))
        })?;
        if root != lease.path() {
            return Err(invalid_call(
                "native private root changed after materialization",
            ));
        }
        for load in &mut self.loads {
            load.library = relative_owned_path(&root, &load.library)?;
            load.manifest = relative_owned_path(&root, &load.manifest)?;
            load.private_root = root.clone();
            load.validate().map_err(invalid_call)?;
        }
        self.lease = Some(lease);
        Ok(())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.loads.is_empty()
    }
}

fn relative_owned_path(root: &Path, path: &Path) -> Result<PathBuf, RuntimeError> {
    let resolved = std::fs::canonicalize(path).map_err(|error| {
        invalid_call(format!("native project artifact is unavailable: {error}"))
    })?;
    let relative = resolved
        .strip_prefix(root)
        .map_err(|_| invalid_call("native project artifact is outside its private root"))?;
    if relative.as_os_str().is_empty()
        || !relative
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
    {
        return Err(invalid_call(
            "native project artifact has an invalid private path",
        ));
    }
    Ok(relative.to_path_buf())
}
