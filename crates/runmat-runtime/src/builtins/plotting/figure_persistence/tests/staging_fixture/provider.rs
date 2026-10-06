use super::*;
use runmat_filesystem::{self as fs, DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
use std::path::{Path, PathBuf};

fn unused() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "unused fixture operation")
}

#[async_trait(?Send)]
impl FsProvider for PausedProvider {
    fn path_namespace(&self) -> fs::FsPathNamespace {
        fs::FsPathNamespace::VirtualRooted
    }
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/"))
    }
    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(self.lease.clone())
    }
    fn begin_staged_replace(&self, _: &Path) -> io::Result<Option<Box<dyn FsStagedOutput>>> {
        Ok(Some(Box::new(PausedStage {
            point: self.point,
            gate: self.gate.lock().unwrap().take(),
            started: self.started.lock().unwrap().take(),
            bytes: Vec::new(),
            published: self.published.clone(),
            drops: self.drops.clone(),
        })))
    }
    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Err(unused())
    }
    async fn read(&self, _: &Path) -> io::Result<Vec<u8>> {
        Err(unused())
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unused())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unused())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(unused())
    }
}
