use std::fs::{self, OpenOptions};
use std::io::Read;
use std::path::Path;

use uuid::Uuid;

use super::store::SupervisorPaths;
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) fn load_or_create_token(paths: &SupervisorPaths) -> NativeExecutionResult<String> {
    let token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    // Clients must never see an empty or partial token. The shared filesystem
    // publishes one complete winner, with private permissions set before writes.
    runmat_filesystem::NativeFsProvider
        .atomic_create_private_new(&paths.token, token.as_bytes(), 64)
        .map_err(protocol_io)?;
    read_token(&paths.token)
}

pub(super) fn read_token(path: &Path) -> NativeExecutionResult<String> {
    validate_private_token_file(path)?;
    let mut token = String::new();
    OpenOptions::new()
        .read(true)
        .open(path)
        .and_then(|file| file.take(65).read_to_string(&mut token))
        .map_err(protocol_io)?;
    if token.len() != 64
        || token
            .bytes()
            .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
    {
        return Err(NativeExecutionError::Protocol(
            "local supervisor token file is malformed".into(),
        ));
    }
    Ok(token)
}

#[cfg(unix)]
fn validate_private_token_file(path: &Path) -> NativeExecutionResult<()> {
    use std::os::unix::fs::PermissionsExt as _;

    let metadata = fs::symlink_metadata(path).map_err(protocol_io)?;
    if !metadata.file_type().is_file() || metadata.permissions().mode() & 0o777 != 0o600 {
        return Err(NativeExecutionError::Configuration(
            "local supervisor token must be a regular mode-0600 file".into(),
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_private_token_file(path: &Path) -> NativeExecutionResult<()> {
    if fs::symlink_metadata(path)
        .map_err(protocol_io)?
        .file_type()
        .is_file()
    {
        Ok(())
    } else {
        Err(NativeExecutionError::Configuration(
            "local supervisor token must be a regular file".into(),
        ))
    }
}

pub(super) fn acquire_supervisor_lock(
    paths: &SupervisorPaths,
) -> NativeExecutionResult<runmat_filesystem::ExclusiveFileLock> {
    runmat_filesystem::ExclusiveFileLock::try_acquire(&paths.lock)
        .map_err(protocol_io)?
        .ok_or_else(|| {
            NativeExecutionError::Configuration(
                "the per-user local execution supervisor already has an authority owner".into(),
            )
        })
}

pub(super) fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        difference |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

fn protocol_io(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests;
