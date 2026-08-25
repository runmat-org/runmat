use std::fs;
use std::path::Path;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(crate) fn prepare_session_root(path: &Path) -> NativeExecutionResult<()> {
    fs::create_dir_all(path).map_err(session_io_error)?;
    let metadata = fs::symlink_metadata(path).map_err(session_io_error)?;
    if !metadata.file_type().is_dir() {
        return Err(NativeExecutionError::Configuration(format!(
            "native session root is not a regular directory: {}",
            path.display()
        )));
    }
    restrict_to_owner(path)?;
    Ok(())
}

#[cfg(unix)]
fn restrict_to_owner(path: &Path) -> NativeExecutionResult<()> {
    use std::os::unix::fs::PermissionsExt as _;

    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(session_io_error)
}

#[cfg(not(unix))]
fn restrict_to_owner(_path: &Path) -> NativeExecutionResult<()> {
    Ok(())
}

fn session_io_error(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(format!("native session store failed: {error}"))
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::fs;

    use super::prepare_session_root;

    #[test]
    fn session_root_is_created_as_a_directory() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("nested").join("session");
        prepare_session_root(&root).unwrap();
        assert!(root.is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn session_root_is_private_and_rejects_symlinks() {
        use std::os::unix::fs::{symlink, PermissionsExt as _};

        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("session");
        prepare_session_root(&root).unwrap();
        assert_eq!(
            fs::metadata(&root).unwrap().permissions().mode() & 0o777,
            0o700
        );

        let target = temporary.path().join("target");
        fs::create_dir(&target).unwrap();
        let link = temporary.path().join("link");
        symlink(&target, &link).unwrap();
        assert!(prepare_session_root(&link).is_err());
    }
}
