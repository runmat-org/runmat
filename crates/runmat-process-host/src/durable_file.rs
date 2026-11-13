//! Publish an already-fsynced staging file with cross-platform replacement.

use std::io;
use std::path::Path;

/// Both paths must share one existing parent. Callers must fsync `staged`
/// before this operation. An error is not a rollback guarantee: the caller
/// must remain fail-closed until it inspects the exact destination.
pub fn publish_synced_file(staged: &Path, destination: &Path) -> io::Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "durable file has no parent"))?;
    if staged.parent() != Some(parent) || !std::fs::metadata(parent)?.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "durable replacement must stay in one directory",
        ));
    }
    match std::fs::symlink_metadata(destination) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "durable destination is not a regular file",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    #[cfg(unix)]
    {
        std::fs::rename(staged, destination)?;
        std::fs::File::open(parent)?.sync_all()?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt as _;
        use windows_sys::Win32::Storage::FileSystem::{
            MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
        };
        let source = staged
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let target = destination
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        if unsafe {
            MoveFileExW(
                source.as_ptr(),
                target.as_ptr(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}
