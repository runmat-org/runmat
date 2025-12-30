use crate::{DirEntry, FsMetadata, FsProvider, OpenFlags};
use std::io::{self, ErrorKind};
use std::path::Path;

#[derive(Default)]
pub struct PlaceholderFsProvider;

impl FsProvider for PlaceholderFsProvider {
    fn open(&self, _path: &Path, _flags: &OpenFlags) -> io::Result<Box<dyn crate::FileHandle>> {
        Err(unsupported())
    }

    fn read(&self, _path: &Path) -> io::Result<Vec<u8>> {
        Err(unsupported())
    }

    fn write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
        Err(unsupported())
    }

    fn remove_file(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    fn metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
        Err(unsupported())
    }

    fn symlink_metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
        Err(unsupported())
    }

    fn read_dir(&self, _path: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unsupported())
    }

    fn canonicalize(&self, _path: &Path) -> io::Result<std::path::PathBuf> {
        Err(unsupported())
    }

    fn create_dir(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    fn create_dir_all(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    fn remove_dir(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    fn remove_dir_all(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    fn rename(&self, _from: &Path, _to: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    fn set_readonly(&self, _path: &Path, _readonly: bool) -> io::Result<()> {
        Err(unsupported())
    }
}

fn unsupported() -> io::Error {
    io::Error::new(
        ErrorKind::Unsupported,
        "filesystem provider not installed for wasm target",
    )
}
