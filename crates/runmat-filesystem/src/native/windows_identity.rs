//! Opened-object identity for scoped Windows workspace authority.
use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
use std::{fs::File, io, os::windows::io::AsRawHandle as _, path::Path};
use windows_sys::Win32::Storage::FileSystem::{
    FileIdInfo, GetFileInformationByHandleEx, FILE_ATTRIBUTE_REPARSE_POINT,
    FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_ID_INFO, FILE_READ_ATTRIBUTES,
};

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    File,
    Directory,
}

pub(crate) fn open(path: &Path, kind: Kind) -> io::Result<File> {
    let file = std::fs::OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)?;
    let metadata = file.metadata()?;
    let expected_kind = match kind {
        Kind::File => metadata.is_file(),
        Kind::Directory => metadata.is_dir(),
    };
    if !expected_kind || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "workspace authority requires the original non-reparse object kind",
        ));
    }
    Ok(file)
}

#[derive(PartialEq, Eq)]
struct Identity {
    volume: u64,
    file: [u8; 16],
}

fn identify(file: &File) -> io::Result<Identity> {
    let mut info = std::mem::MaybeUninit::<FILE_ID_INFO>::uninit();
    // SAFETY: file owns a live handle; the output has the exact API structure
    // size and is read only after the call reports complete initialization.
    if unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle(),
            FileIdInfo,
            info.as_mut_ptr().cast(),
            std::mem::size_of::<FILE_ID_INFO>() as u32,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: successful GetFileInformationByHandleEx initialized the structure.
    let info = unsafe { info.assume_init() };
    Ok(Identity {
        volume: info.VolumeSerialNumber,
        file: info.FileId.Identifier,
    })
}

pub(crate) fn same(left: &File, right: &File) -> io::Result<bool> {
    Ok(identify(left)? == identify(right)?)
}
