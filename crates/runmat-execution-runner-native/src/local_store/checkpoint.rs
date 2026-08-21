use std::fs;
use std::path::PathBuf;

use runmat_execution_runner::DriverSnapshot;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(crate) struct CheckpointStore {
    root: PathBuf,
}

impl CheckpointStore {
    pub(crate) fn new(root: PathBuf) -> NativeExecutionResult<Self> {
        fs::create_dir_all(&root).map_err(io_error)?;
        Ok(Self { root })
    }

    pub(crate) fn write(&self, snapshot: &DriverSnapshot) -> NativeExecutionResult<()> {
        let payload = serde_json::to_vec(snapshot)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let next = self.root.join("driver.json.next");
        let current = self.root.join("driver.json");
        fs::write(&next, payload).map_err(io_error)?;
        fs::rename(next, current).map_err(io_error)
    }
}

fn io_error(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(format!("local checkpoint store failed: {error}"))
}
