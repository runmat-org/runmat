//! Point-in-time sandbox authorization before acquiring the shared native stage.
//! The sandbox provider's ordinary path operations are path-based too; this
//! does not claim descriptor-relative confinement against concurrent renames.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Component, Path, PathBuf};

pub(super) fn authorized_destination(root: &Path, resolved: &Path) -> io::Result<PathBuf> {
    let relative = resolved.strip_prefix(root).map_err(|_| outside_root())?;
    let leaf = relative.file_name().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidInput,
            "sandbox root is not a file destination",
        )
    })?;
    let mut parent = root.to_path_buf();
    for component in relative
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .components()
    {
        let Component::Normal(segment) = component else {
            return Err(outside_root());
        };
        let next = parent.join(segment);
        match fs::create_dir(&next) {
            Ok(()) => {}
            Err(source) if source.kind() == ErrorKind::AlreadyExists => {}
            Err(source) => return Err(source),
        }
        let canonical = fs::canonicalize(&next)?;
        if !canonical.starts_with(root) {
            return Err(outside_root());
        }
        if !fs::metadata(&canonical)?.is_dir() {
            return Err(io::Error::new(
                ErrorKind::NotADirectory,
                "sandbox staged destination parent is not a directory",
            ));
        }
        parent = canonical;
    }
    Ok(parent.join(leaf))
}

fn outside_root() -> io::Error {
    io::Error::new(
        ErrorKind::PermissionDenied,
        "sandbox staged destination resolves outside its root",
    )
}
