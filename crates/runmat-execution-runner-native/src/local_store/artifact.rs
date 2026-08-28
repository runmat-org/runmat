use std::fs;
use std::path::PathBuf;

use runmat_execution::identity::ArtifactId;
use runmat_execution::Digest;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(crate) struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    pub(crate) fn new(root: PathBuf) -> NativeExecutionResult<Self> {
        fs::create_dir_all(&root).map_err(io_error)?;
        Ok(Self { root })
    }

    pub(crate) fn put(&self, id: ArtifactId, bytes: &[u8]) -> NativeExecutionResult<()> {
        let path = self.root.join(id.to_string());
        if !path.exists() {
            fs::write(&path, bytes).map_err(io_error)?;
        }
        if Digest::sha256(&fs::read(&path).map_err(io_error)?) != Digest::sha256(bytes) {
            return Err(NativeExecutionError::Protocol(
                "local artifact store identity collision".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn get(&self, id: ArtifactId) -> NativeExecutionResult<Vec<u8>> {
        fs::read(self.root.join(id.to_string())).map_err(io_error)
    }
}

fn io_error(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(format!("local artifact store failed: {error}"))
}
