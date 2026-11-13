use std::path::Path;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(crate) fn prepare_session_root(path: &Path) -> NativeExecutionResult<()> {
    // The session may create its supervisor parent before durable bootstrap.
    // Use the shared owner for the entire new directory chain, not only the leaf.
    runmat_filesystem::NativeFsProvider
        .create_private_directory_durable(path)
        .map_err(session_io_error)
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
        let parent = temporary.path().join("authority");
        let root = parent.join("sessions");
        prepare_session_root(&root).unwrap();
        for directory in [&parent, &root] {
            assert_eq!(
                fs::metadata(directory).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }

        let target = temporary.path().join("target");
        fs::create_dir(&target).unwrap();
        let link = temporary.path().join("link");
        symlink(&target, &link).unwrap();
        assert!(prepare_session_root(&link).is_err());
    }
}
