use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Serialize;
use uuid::Uuid;

use crate::{NativeExecutionError, NativeExecutionResult};

const MAX_PERSISTED_JSON_BYTES: u64 = 64 * 1024 * 1024;

pub(super) fn atomic_json(path: &Path, value: &impl Serialize) -> NativeExecutionResult<()> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    atomic_write(path, &bytes)
}

pub(super) fn read_json<T: DeserializeOwned>(path: &Path) -> NativeExecutionResult<T> {
    let mut file = File::open(path).map_err(io_error)?;
    let length = file.metadata().map_err(io_error)?.len();
    if length > MAX_PERSISTED_JSON_BYTES {
        return Err(NativeExecutionError::Protocol(format!(
            "{} exceeds the durable metadata size limit",
            path.display()
        )));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    use std::io::Read as _;
    file.read_to_end(&mut bytes).map_err(io_error)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| NativeExecutionError::Protocol(format!("{}: {error}", path.display())))
}

pub(super) fn atomic_write(path: &Path, bytes: &[u8]) -> NativeExecutionResult<()> {
    let parent = path.parent().ok_or_else(|| {
        NativeExecutionError::Configuration("durable metadata path has no parent".into())
    })?;
    secure_directory(parent)?;
    let temporary = parent.join(format!(".write-{}.tmp", Uuid::new_v4().simple()));
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary).map_err(io_error)?;
    file.write_all(bytes).map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    replace_file(&temporary, path)?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(io_error)?;
    Ok(())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> NativeExecutionResult<()> {
    fs::rename(source, destination).map_err(io_error)
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> NativeExecutionResult<()> {
    use std::os::windows::ffi::OsStrExt as _;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let mut source = source.as_os_str().encode_wide().collect::<Vec<_>>();
    source.push(0);
    let mut destination = destination.as_os_str().encode_wide().collect::<Vec<_>>();
    destination.push(0);
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(io_error(std::io::Error::last_os_error()))
    } else {
        Ok(())
    }
}

pub(super) fn secure_directory(path: &Path) -> NativeExecutionResult<()> {
    fs::create_dir_all(path).map_err(io_error)?;
    let metadata = fs::symlink_metadata(path).map_err(io_error)?;
    if !metadata.file_type().is_dir() {
        return Err(NativeExecutionError::Configuration(format!(
            "durable state path is not a regular directory: {}",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(io_error)?;
    }
    Ok(())
}

pub(super) fn io_error(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

pub(super) fn unix_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}
