use runmat_filesystem as vfs;
use std::io;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(super) struct MovePlanEntry {
    pub(super) source: PathBuf,
    pub(super) source_display: String,
    pub(super) target: PathBuf,
    pub(super) target_display: String,
    pub(super) replace_directory: Option<bool>,
}

pub(super) struct MoveFailure {
    pub(super) source: String,
    pub(super) target: String,
    pub(super) error: io::Error,
}

pub(super) async fn execute(entries: &[MovePlanEntry]) -> Result<(), MoveFailure> {
    for entry in entries {
        if let Some(directory) = entry.replace_directory {
            if let Err(error) =
                super::super::target::remove_existing(&entry.target, directory).await
            {
                return Err(failure(entry, error));
            }
        }
        if let Err(error) = vfs::rename_async(&entry.source, &entry.target).await {
            return Err(failure(entry, error));
        }
    }
    Ok(())
}

fn failure(entry: &MovePlanEntry, error: io::Error) -> MoveFailure {
    MoveFailure {
        source: entry.source_display.clone(),
        target: entry.target_display.clone(),
        error,
    }
}
