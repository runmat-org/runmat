use crate::{DirEntry, FileHandle, FsFileType, FsMetadata, FsProvider, OpenFlags};
use std::fs;
use std::io;
use std::path::Path;

#[derive(Default)]
pub struct NativeFsProvider;

impl FsProvider for NativeFsProvider {
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let mut opts = fs::OpenOptions::new();
        opts.read(flags.read);
        opts.write(flags.write);
        opts.append(flags.append);
        opts.truncate(flags.truncate);
        opts.create(flags.create);
        opts.create_new(flags.create_new);
        let file = opts.open(path)?;
        Ok(Box::new(file))
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        fs::read(path)
    }

    fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        fs::write(path, data)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }

    fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        fs::metadata(path).map(FsMetadata::from)
    }

    fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        fs::symlink_metadata(path).map(FsMetadata::from)
    }

    fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let entries = fs::read_dir(path)?;
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry?;
            let file_type = entry
                .file_type()
                .ok()
                .map(FsFileType::from)
                .unwrap_or(FsFileType::Unknown);
            out.push(DirEntry {
                path: entry.path(),
                file_name: entry.file_name(),
                file_type,
            });
        }
        Ok(out)
    }

    fn canonicalize(&self, path: &Path) -> io::Result<std::path::PathBuf> {
        fs::canonicalize(path)
    }

    fn create_dir(&self, path: &Path) -> io::Result<()> {
        fs::create_dir(path)
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        fs::create_dir_all(path)
    }

    fn remove_dir(&self, path: &Path) -> io::Result<()> {
        fs::remove_dir(path)
    }

    fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        fs::remove_dir_all(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        fs::rename(from, to)
    }

    fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        let mut perms = fs::metadata(path)?.permissions();
        perms.set_readonly(readonly);
        fs::set_permissions(path, perms)
    }
}

impl From<fs::Metadata> for FsMetadata {
    fn from(meta: fs::Metadata) -> Self {
        let file_type = FsFileType::from(meta.file_type());
        FsMetadata {
            file_type,
            len: meta.len(),
            modified: meta.modified().ok(),
            readonly: meta.permissions().readonly(),
            hash: None,
        }
    }
}

impl From<fs::FileType> for FsFileType {
    fn from(ft: fs::FileType) -> Self {
        if ft.is_dir() {
            FsFileType::Directory
        } else if ft.is_file() {
            FsFileType::File
        } else if ft.is_symlink() {
            FsFileType::Symlink
        } else {
            FsFileType::Other
        }
    }
}
