//! Copy-on-write storage beneath the dataset manifest publication boundary.

use super::{arrays_root, data_error, data_error_with_source};
use crate::BuiltinResult;
use runmat_filesystem::FsProvider;
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};

pub(crate) struct ChunkStorageIdentity {
    pub hash: String,
    pub object_id: String,
    pub file_name: String,
}

impl ChunkStorageIdentity {
    pub fn for_payload(bytes: &[u8]) -> Self {
        let digest = hex::encode(Sha256::digest(bytes));
        let hash = format!("sha256:{digest}");
        let object_id = format!("obj_{hash}");
        let file_name = format!("obj_{digest}.json");
        Self {
            hash,
            object_id,
            file_name,
        }
    }
}

/// Dataset metadata uses slash-separated relative paths on every host.
pub(crate) fn relative_storage_path(root: &Path, file: &Path) -> BuiltinResult<String> {
    let relative = file.strip_prefix(root).map_err(|cause| {
        data_error_with_source("failed to compute dataset-relative storage path", cause)
    })?;
    let mut encoded = String::with_capacity(relative.as_os_str().len());
    for component in relative.components() {
        match component {
            Component::Normal(name) => {
                let name = name.to_str().ok_or_else(|| {
                    data_error("dataset storage paths require Unicode components")
                })?;
                if !encoded.is_empty() {
                    encoded.push('/');
                }
                encoded.push_str(name);
            }
            Component::CurDir => {}
            _ => return Err(data_error("dataset storage paths must remain relative")),
        }
    }
    if encoded.is_empty() {
        return Err(data_error("dataset storage path must name an object"));
    }
    Ok(encoded)
}

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
    // Revision names are opaque random identities, encoded as portable path
    // components. The public content-hash spelling contains a Windows colon.
    let identity = hex::encode(nonce);
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
