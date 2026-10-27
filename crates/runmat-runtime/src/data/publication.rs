//! Copy-on-write storage beneath the dataset manifest publication boundary.

use super::{arrays_root, data_error_with_source};
use crate::BuiltinResult;
use runmat_filesystem::FsProvider;
use std::path::{Path, PathBuf};

/// Every mutation writes a fresh revision. Only the manifest makes it reachable;
/// a rejected publication must not change bytes reachable from an older manifest.
/// Unreferenced revisions are retained rather than deleting objects that another
/// manifest, snapshot or reader may still own.
pub(crate) async fn create_array_revision(
    provider: &dyn FsProvider,
    root: &Path,
    array: &str,
) -> BuiltinResult<PathBuf> {
    let mut nonce = [0u8; 32];
    getrandom::getrandom(&mut nonce).map_err(|error| {
        data_error_with_source("failed to allocate dataset revision identity", error)
    })?;
    let identity = super::sha256_hex(&nonce);
    let path = arrays_root(root)
        .join(array)
        .join("revisions")
        .join(identity);
    provider.create_dir_all(&path).await.map_err(|error| {
        data_error_with_source(
            format!(
                "failed to create array revision '{}': {error}",
                path.display()
            ),
            error,
        )
    })?;
    Ok(path)
}
