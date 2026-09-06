use std::io;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(super) struct CopyPlanEntry {
    pub(super) source: PathBuf,
    pub(super) source_display: String,
    pub(super) target: PathBuf,
    pub(super) target_display: String,
    pub(super) source_is_directory: bool,
    pub(super) replace: Option<TargetKind>,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum TargetKind {
    File,
    Directory,
}

pub(super) struct CopyFailure {
    pub(super) source: String,
    pub(super) target: String,
    pub(super) error: io::Error,
}

pub(super) async fn execute(entries: &[CopyPlanEntry]) -> Result<(), CopyFailure> {
    for entry in entries {
        if let Some(kind) = entry.replace {
            if let Err(error) = super::super::target::remove_existing(
                &entry.target,
                matches!(kind, TargetKind::Directory),
            )
            .await
            {
                return Err(failure(entry, error));
            }
        }
        if let Err(error) =
            super::filesystem::copy(&entry.source, &entry.target, entry.source_is_directory).await
        {
            return Err(failure(entry, error));
        }
    }
    Ok(())
}

fn failure(entry: &CopyPlanEntry, error: io::Error) -> CopyFailure {
    CopyFailure {
        source: entry.source_display.clone(),
        target: entry.target_display.clone(),
        error,
    }
}
