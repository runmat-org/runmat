//! Native snapshot storage delegates atomic replacement to the shared filesystem.
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution_runner::DriverSnapshot;
use runmat_filesystem::{
    AtomicReplaceOutcome, FsProvider, NativeFsProvider, PublicationDurability,
};
use std::{fs, path::PathBuf};

pub(crate) struct CheckpointStore {
    root: PathBuf,
}

impl CheckpointStore {
    pub(crate) fn new(root: PathBuf) -> NativeExecutionResult<Self> {
        fs::create_dir_all(&root).map_err(io_error)?;
        Ok(Self { root })
    }

    pub(crate) fn write(
        &self,
        snapshot: &DriverSnapshot,
    ) -> NativeExecutionResult<AtomicReplaceOutcome> {
        snapshot.validate()?;
        let payload = serde_json::to_vec(snapshot)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
        let mut outcome = futures::executor::block_on(
            NativeFsProvider.atomic_replace(&self.root.join("driver.json"), &payload),
        )
        .map_err(io_error)?;
        if outcome.durability == PublicationDurability::Durable {
            // Session construction may have created several fresh directories.
            // Flush their namespace too, rather than inferring durable ancestors
            // from the final checkpoint directory's successful flush.
            for ancestor in self.root.ancestors().filter(|p| !p.as_os_str().is_empty()) {
                if let Err(error) =
                    fs::File::open(ancestor).and_then(|directory| directory.sync_all())
                {
                    outcome.durability = PublicationDurability::AtomicVisibility;
                    outcome.durability_diagnostic = Some(format!(
                        "checkpoint namespace flush failed at {}: {error}",
                        ancestor.display()
                    ));
                    break;
                }
            }
        }
        Ok(outcome)
    }
}

fn io_error(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(format!("local checkpoint store did not publish: {error}"))
}

#[cfg(test)]
mod tests;
