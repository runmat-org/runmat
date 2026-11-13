//! Directory-entry durability shared by native publication owners.
use std::{io, path::Path};

pub(crate) fn sync(path: &Path) -> io::Result<()> {
    let path = if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    };
    #[cfg(unix)]
    {
        std::fs::File::open(path)?.sync_all()
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
        use windows_sys::Win32::{
            Foundation::GENERIC_WRITE,
            Storage::FileSystem::{
                FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS,
                FILE_FLAG_OPEN_REPARSE_POINT, FILE_READ_ATTRIBUTES,
            },
        };
        // FlushFileBuffers requires write access. Open the directory itself,
        // not a reparse target, and validate that same handle before flushing.
        let directory = std::fs::OpenOptions::new()
            .access_mode(GENERIC_WRITE | FILE_READ_ATTRIBUTES)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)?;
        let metadata = directory.metadata()?;
        if !metadata.is_dir() || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "directory synchronization requires a non-reparse directory",
            ));
        }
        directory.sync_all()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native directory synchronization is unavailable on this platform",
        ))
    }
}
