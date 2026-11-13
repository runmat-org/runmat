//! Compatible native replacement without deleting a destination to emulate it.
use std::{fs, io, path::Path};

pub(crate) fn replace(from: &Path, to: &Path) -> io::Result<()> {
    let source = fs::symlink_metadata(from)?;
    match fs::symlink_metadata(to) {
        Ok(destination) if source.is_dir() != destination.is_dir() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "replacement source and destination have incompatible object kinds",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    // In particular, Windows rename may replace a file with a directory. Reject
    // that incompatible request above; never remove a winner and retry a move.
    // Other rename restrictions, including nonempty directories, stay with OS.
    fs::rename(from, to)
}
