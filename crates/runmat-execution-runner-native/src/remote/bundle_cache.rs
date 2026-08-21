use std::io::Write as _;
use std::path::{Path, PathBuf};

use runmat_execution::Digest;
use runmat_execution_artifact::archive::{read_bundle, ArchiveLimits};
use runmat_execution_artifact::ExecutionBundle;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) fn store(cache: &Path, digest: Digest, bytes: &[u8]) -> NativeExecutionResult<()> {
    create_private_directory(cache)?;
    if Digest::sha256(bytes) != digest {
        return Err(protocol("bundle cache input does not match its digest"));
    }
    let target = path(cache, digest);
    if target.exists() {
        return validate_existing(&target, digest);
    }

    let temporary = cache.join(format!(".{digest}.{}.tmp", uuid::Uuid::new_v4()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary).map_err(protocol)?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        let _ = std::fs::remove_file(&temporary);
        return Err(protocol(error));
    }
    drop(file);

    match std::fs::rename(&temporary, &target) {
        Ok(()) => validate_existing(&target, digest),
        Err(_) if target.exists() => {
            let _ = std::fs::remove_file(&temporary);
            validate_existing(&target, digest)
        }
        Err(error) => {
            let _ = std::fs::remove_file(&temporary);
            Err(protocol(error))
        }
    }
}

pub(super) fn load(cache: &Path, digest: Digest) -> NativeExecutionResult<(ExecutionBundle, u64)> {
    let target = path(cache, digest);
    validate_existing(&target, digest)?;
    let bytes = std::fs::read(target).map_err(protocol)?;
    let stored_bytes = bytes.len() as u64;
    let bundle = read_bundle(bytes.as_slice(), ArchiveLimits::default()).map_err(protocol)?;
    Ok((bundle, stored_bytes))
}

fn path(cache: &Path, digest: Digest) -> PathBuf {
    cache.join(format!("{digest}.rmbundle"))
}

fn validate_existing(target: &Path, digest: Digest) -> NativeExecutionResult<()> {
    let bytes = std::fs::read(target).map_err(protocol)?;
    if Digest::sha256(&bytes) != digest {
        return Err(protocol("node bundle cache contains substituted bytes"));
    }
    Ok(())
}

fn create_private_directory(path: &Path) -> NativeExecutionResult<()> {
    std::fs::create_dir_all(path).map_err(protocol)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).map_err(protocol)?;
    }
    Ok(())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_rejects_substitution_and_leaves_no_temporary_files() {
        let directory = tempfile::tempdir().unwrap();
        let bytes = b"not an archive";
        let digest = Digest::sha256(bytes);
        store(directory.path(), digest, bytes).unwrap();
        store(directory.path(), digest, bytes).unwrap();
        assert_eq!(
            std::fs::read(path(directory.path(), digest)).unwrap(),
            bytes
        );
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);

        std::fs::write(path(directory.path(), digest), b"substituted").unwrap();
        assert!(store(directory.path(), digest, bytes).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn cache_is_private_on_unix() {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().unwrap();
        let digest = Digest::sha256(b"exact bytes");
        store(directory.path(), digest, b"exact bytes").unwrap();
        assert_eq!(
            std::fs::metadata(directory.path())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(path(directory.path(), digest))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}
