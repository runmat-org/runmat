//! Task-state leaf protection, not an authenticated/hostile-parent namespace.
use super::*;
use std::path::Component;
use windows_sys::Win32::Foundation::ERROR_ALREADY_EXISTS;
use windows_sys::Win32::Storage::FileSystem::{
    CreateDirectoryW, FILE_FLAG_BACKUP_SEMANTICS, FILE_READ_ATTRIBUTES, READ_CONTROL,
};

#[derive(Clone, Copy)]
pub(in crate::native) enum Mode {
    Existing,
    CreateMissing,
    CreateMissingDurable,
}

pub(in crate::native) fn open(path: &Path, mode: Mode) -> io::Result<File> {
    check_chain(path, mode)?;
    let file = open_leaf(path)?;
    if matches!(mode, Mode::CreateMissingDurable) {
        crate::native::directory_sync::sync(path)?;
    }
    Ok(file)
}

fn open_leaf(path: &Path) -> io::Result<File> {
    let encoded = wide(path)?;
    // SAFETY: valid path. Keep a leaf handle without delete sharing through
    // credential load/publication. This is deliberately not an ancestry lock.
    let handle = unsafe {
        CreateFileW(
            encoded.as_ptr(),
            READ_CONTROL | FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            null_mut(),
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
            null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: one uniquely owned valid CreateFileW handle.
    let file = unsafe { File::from_raw_handle(handle) };
    verify(&file, PrivateObject::Directory)?;
    Ok(file)
}

pub(super) fn check_chain(path: &Path, mode: Mode) -> io::Result<()> {
    if !path.is_absolute() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "private Windows state path must be absolute",
        ));
    }
    let mut current = std::path::PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) => {
                current.push(component.as_os_str());
                continue;
            }
            Component::RootDir | Component::Normal(_) => current.push(component.as_os_str()),
            Component::CurDir | Component::ParentDir => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    "private Windows state path must not contain relative components",
                ))
            }
        }
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) => check_observed_directory(&metadata)?,
            Err(error)
                if matches!(mode, Mode::CreateMissing | Mode::CreateMissingDurable)
                    && error.kind() == ErrorKind::NotFound =>
            {
                let descriptor = descriptor(PrivateObject::Directory)?;
                let attributes = attributes(&descriptor);
                let encoded = wide(&current)?;
                // SAFETY: protected ACL is supplied before a new directory is
                // visible. Existing ancestors' ACLs/ownership are never reset.
                if unsafe { CreateDirectoryW(encoded.as_ptr(), &attributes) } == 0 {
                    let error = io::Error::last_os_error();
                    if error.raw_os_error() != Some(ERROR_ALREADY_EXISTS as i32) {
                        return Err(error);
                    }
                }
                check_observed_directory(&std::fs::symlink_metadata(&current)?)?;
                // Validate a race winner too; never repair a foreign ACL.
                let _guard = open_leaf(&current)?;
                if matches!(mode, Mode::CreateMissingDurable) {
                    crate::native::directory_sync::sync(&current)?;
                    crate::native::directory_sync::sync(
                        current
                            .parent()
                            .ok_or_else(|| invalid("directory needs a parent"))?,
                    )?;
                }
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn check_observed_directory(metadata: &std::fs::Metadata) -> io::Result<()> {
    if !metadata.is_dir() || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(invalid(
            "private Windows state path contains a non-directory or reparse point",
        ));
    }
    Ok(())
}
