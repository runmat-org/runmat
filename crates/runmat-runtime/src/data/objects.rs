//! Dataset objects are immutable. Only the manifest publishes a new dataset
//! revision; failed publication must not change objects used by older readers.
use std::path::{Path, PathBuf};

use super::{data_error, sha256_hex};
use crate::BuiltinResult;
use runmat_filesystem::{self as fs, AtomicCreateResult};

pub async fn write_object(root: &Path, bytes: &[u8]) -> BuiltinResult<PathBuf> {
    publish(
        root.join("objects")
            .join(format!("{}.json", sha256_hex(bytes))),
        bytes,
    )
    .await
}

pub async fn write_chunk_index(root: &Path, array: &str, bytes: &[u8]) -> BuiltinResult<PathBuf> {
    // Keep the existing array/index layout understood by chunk retention. Each
    // index revision has its own immutable path; the manifest switches the head.
    publish(
        root.join("arrays")
            .join(array)
            .join("revisions")
            .join(sha256_hex(bytes))
            .join("chunks/index.json"),
        bytes,
    )
    .await
}

async fn publish(path: PathBuf, bytes: &[u8]) -> BuiltinResult<PathBuf> {
    fs::create_dir_all_async(path.parent().expect("dataset object parent"))
        .await
        .map_err(|error| data_error(format!("cannot create dataset object directory: {error}")))?;
    match fs::atomic_create_new_async(&path, bytes)
        .await
        .map_err(|error| {
            data_error(format!(
                "cannot publish dataset object '{}': {error}",
                path.display()
            ))
        })? {
        AtomicCreateResult::Published { .. } => Ok(path),
        AtomicCreateResult::Existing(existing) if existing == bytes => Ok(path),
        AtomicCreateResult::Existing(_) => Err(data_error(format!(
            "dataset object integrity mismatch at '{}'",
            path.display()
        ))),
    }
}
