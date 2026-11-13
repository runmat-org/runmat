use std::io;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
pub(super) enum DestinationPolicy {
    New,
    ReplaceCompatible,
}

#[derive(Debug, Clone)]
pub(super) struct MovePlanEntry {
    pub(super) source: PathBuf,
    pub(super) source_display: String,
    pub(super) target: PathBuf,
    pub(super) target_display: String,
    pub(super) destination_policy: DestinationPolicy,
}

pub(super) struct MoveFailure {
    pub(super) source: String,
    pub(super) target: String,
    pub(super) error: io::Error,
}

pub(super) async fn execute(
    fs: &mut super::super::authority::TransferFilesystem,
    entries: &[MovePlanEntry],
) -> crate::BuiltinResult<Result<(), MoveFailure>> {
    for entry in entries {
        fs.checkpoint().await?;
        if fs.same_path(&entry.source, &entry.target).await {
            continue;
        }
        if fs.is_descendant(&entry.source, &entry.target).await {
            return Ok(Err(failure(
                entry,
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "cannot move a directory into its own descendant",
                ),
            )));
        }
        let result = match entry.destination_policy {
            DestinationPolicy::New => fs.rename(&entry.source, &entry.target).await,
            DestinationPolicy::ReplaceCompatible => {
                fs.rename_replace(&entry.source, &entry.target).await
            }
        };
        if let Err(error) = result {
            // The provider may complete at the same instant cancellation wins.
            // Resolve invocation cancellation before converting an ordinary
            // filesystem failure into movefile's status-zero result.
            fs.checkpoint().await?;
            return Ok(Err(failure(entry, error)));
        }
    }
    Ok(Ok(()))
}

fn failure(entry: &MovePlanEntry, error: io::Error) -> MoveFailure {
    MoveFailure {
        source: entry.source_display.clone(),
        target: entry.target_display.clone(),
        error,
    }
}
