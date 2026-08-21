use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use runmat_process_host::is_process_alive;
use uuid::Uuid;

use super::store::SupervisorPaths;
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) fn load_or_create_token(paths: &SupervisorPaths) -> NativeExecutionResult<String> {
    if paths.token.exists() {
        return read_token(&paths.token);
    }
    let token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    match options.open(&paths.token) {
        Ok(mut file) => {
            file.write_all(token.as_bytes()).map_err(protocol_io)?;
            file.sync_all().map_err(protocol_io)?;
            Ok(token)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => read_token(&paths.token),
        Err(error) => Err(protocol_io(error)),
    }
}

pub(super) fn read_token(path: &Path) -> NativeExecutionResult<String> {
    validate_private_token_file(path)?;
    let mut token = String::new();
    OpenOptions::new()
        .read(true)
        .open(path)
        .and_then(|mut file| file.read_to_string(&mut token))
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

pub(super) struct SupervisorLock {
    path: PathBuf,
    process_id: u32,
}

impl SupervisorLock {
    pub(super) fn acquire(paths: &SupervisorPaths) -> NativeExecutionResult<Self> {
        let process_id = std::process::id();
        for _ in 0..2 {
            let mut options = OpenOptions::new();
            options.create_new(true).write(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt as _;
                options.mode(0o600);
            }
            match options.open(&paths.lock) {
                Ok(mut file) => {
                    writeln!(file, "{process_id}").map_err(protocol_io)?;
                    file.sync_all().map_err(protocol_io)?;
                    return Ok(Self {
                        path: paths.lock.clone(),
                        process_id,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let owner = fs::read_to_string(&paths.lock)
                        .ok()
                        .and_then(|value| value.trim().parse::<u32>().ok());
                    if owner.is_some_and(is_process_alive) {
                        return Err(NativeExecutionError::Configuration(
                            "the per-user local execution supervisor is already running".into(),
                        ));
                    }
                    fs::remove_file(&paths.lock).map_err(protocol_io)?;
                }
                Err(error) => return Err(protocol_io(error)),
            }
        }
        Err(NativeExecutionError::Protocol(
            "unable to acquire the local supervisor lock".into(),
        ))
    }
}

impl Drop for SupervisorLock {
    fn drop(&mut self) {
        let owner = fs::read_to_string(&self.path)
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok());
        if owner == Some(self.process_id) {
            let _ = fs::remove_file(&self.path);
        }
    }
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
