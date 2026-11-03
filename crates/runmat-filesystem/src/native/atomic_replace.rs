//! One native atomic-replacement primitive shared by storage consumers.
use crate::{AtomicReplaceOutcome, PublicationDurability};
use std::{
    fs, io,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Debug)]
enum Operation {
    InspectParent,
    InspectDestination,
    CreateStage,
    WriteStage,
    FlushStage,
    Rename,
}

/// Private context for this primitive. Keep the OS error in the source chain;
/// callers still receive its original ErrorKind and publication semantics.
#[derive(Debug)]
struct ReplacementFailure {
    operation: Operation,
    path: PathBuf,
    source: io::Error,
    cleanup: Option<io::Error>,
}

impl ReplacementFailure {
    fn new(operation: Operation, path: &Path, source: io::Error) -> Self {
        Self {
            operation,
            path: path.to_owned(),
            source,
            cleanup: None,
        }
    }
    fn into_io(self) -> io::Error {
        io::Error::new(self.source.kind(), self)
    }
}
impl std::fmt::Display for ReplacementFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "atomic replacement {:?} at {}: {}",
            self.operation,
            self.path.display(),
            self.source
        )?;
        if let Some(cleanup) = &self.cleanup {
            write!(f, "; staging cleanup failed: {cleanup}")?;
        }
        Ok(())
    }
}
impl std::error::Error for ReplacementFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

pub(super) fn publish(path: &Path, bytes: &[u8]) -> io::Result<AtomicReplaceOutcome> {
    publish_with_directory_sync(path, bytes, super::directory_sync::sync)
}

fn publish_with_directory_sync(
    path: &Path,
    bytes: &[u8],
    sync_directory: impl FnOnce(&Path) -> io::Result<()>,
) -> io::Result<AtomicReplaceOutcome> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "atomic replacement requires a parent directory",
            )
        })?;
    // Do not create an unflushed ancestor chain and then claim durable publication.
    // The consumer owns directory creation and its retention policy.
    if !fs::metadata(parent)
        .map_err(|error| {
            ReplacementFailure::new(Operation::InspectParent, parent, error).into_io()
        })?
        .is_dir()
    {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            "replacement parent is not a directory",
        ));
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.is_file() || metadata.file_type().is_symlink() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "replacement destination is not a regular file",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(
                ReplacementFailure::new(Operation::InspectDestination, path, error).into_io(),
            )
        }
    }
    let staged = parent.join(format!(
        ".replace-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&staged).map_err(|error| {
        ReplacementFailure::new(Operation::CreateStage, &staged, error).into_io()
    })?;
    let preparation = file
        .write_all(bytes)
        .map_err(|error| ReplacementFailure::new(Operation::WriteStage, &staged, error))
        .and_then(|()| {
            file.sync_all()
                .map_err(|error| ReplacementFailure::new(Operation::FlushStage, &staged, error))
        });
    drop(file);
    let publication = preparation.and_then(|()| {
        fs::rename(&staged, path)
            .map_err(|error| ReplacementFailure::new(Operation::Rename, path, error))
    });
    if let Err(mut error) = publication {
        if let Err(cleanup) = fs::remove_file(&staged) {
            error.cleanup = Some(cleanup);
        }
        return Err(error.into_io());
    }
    // Rename already committed visibility. A later flush error must not be
    // reported as an uncommitted write that callers can roll back or repeat.
    Ok(match sync_directory(parent) {
        Ok(()) => AtomicReplaceOutcome {
            durability: PublicationDurability::Durable,
            durability_diagnostic: None,
        },
        Err(error) => AtomicReplaceOutcome {
            durability: PublicationDurability::AtomicVisibility,
            durability_diagnostic: Some(error.to_string()),
        },
    })
}

#[cfg(test)]
mod tests;
