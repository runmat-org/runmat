use std::path::PathBuf;

use runmat_execution::identity::ArtifactId;
use runmat_filesystem::{AtomicCreateResult, NativeFsProvider};

use crate::{NativeExecutionError, NativeExecutionResult};

pub(crate) struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    pub(crate) fn new(root: PathBuf) -> NativeExecutionResult<Self> {
        NativeFsProvider
            .create_private_directory_durable(&root)
            .map_err(io_error)?;
        Ok(Self { root })
    }

    pub(crate) fn put(&self, id: ArtifactId, bytes: &[u8]) -> NativeExecutionResult<()> {
        let path = self.root.join(id.to_string());
        // Publish one complete winner. The shared filesystem owns no-replace
        // linearization; checking existence before an ordinary write does not.
        let publication = NativeFsProvider
            .atomic_create_new_sync(&path, bytes)
            .map_err(io_error)?;
        if let AtomicCreateResult::Existing(existing) = publication {
            if existing != bytes {
                return Err(NativeExecutionError::Protocol(
                    "local artifact store identity collision".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn get(&self, id: ArtifactId, maximum_bytes: u64) -> NativeExecutionResult<Vec<u8>> {
        NativeFsProvider
            .read_bounded_sync(&self.root.join(id.to_string()), maximum_bytes)
            .map_err(io_error)
    }
}

fn io_error(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(format!("local artifact store failed: {error}"))
}

#[cfg(test)]
mod tests;
