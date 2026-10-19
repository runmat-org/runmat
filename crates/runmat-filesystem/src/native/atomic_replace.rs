//! One native atomic-replacement primitive shared by storage consumers.
use crate::{AtomicReplaceOutcome, PublicationDurability};
use std::{fs, io, io::Write, path::Path};

pub(super) fn publish(path: &Path, bytes: &[u8]) -> io::Result<AtomicReplaceOutcome> {
    publish_with_directory_sync(path, bytes, |parent| fs::File::open(parent)?.sync_all())
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
    if !fs::metadata(parent)?.is_dir() {
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
        Err(error) => return Err(error),
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
    let mut file = options.open(&staged)?;
    let preparation = file.write_all(bytes).and_then(|()| file.sync_all());
    drop(file);
    let publication = preparation.and_then(|()| fs::rename(&staged, path));
    if let Err(error) = publication {
        return match fs::remove_file(&staged) {
            Ok(()) => Err(error),
            Err(cleanup) => Err(io::Error::new(
                error.kind(),
                format!("{error}; staging cleanup failed: {cleanup}"),
            )),
        };
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
