//! Native snapshot storage delegates atomic replacement to the shared filesystem.
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution_runner::DriverSnapshot;
use runmat_filesystem::{AtomicReplaceOutcome, NativeFsProvider};
use std::path::PathBuf;

pub(crate) struct CheckpointStore {
    root: PathBuf,
}

impl CheckpointStore {
    pub(crate) fn new(root: PathBuf) -> NativeExecutionResult<Self> {
        NativeFsProvider
            .create_private_directory_durable(&root)
            .map_err(io_error)?;
        Ok(Self { root })
    }

    pub(crate) fn write(
        &self,
        snapshot: &DriverSnapshot,
    ) -> NativeExecutionResult<AtomicReplaceOutcome> {
        snapshot.validate()?;
        let payload = serde_json::to_vec(snapshot)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        NativeFsProvider
            .atomic_replace_sync(&self.root.join("driver.json"), &payload)
            .map_err(io_error)
    }
}

fn io_error(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(format!("local checkpoint store did not publish: {error}"))
}

#[cfg(test)]
mod tests;
