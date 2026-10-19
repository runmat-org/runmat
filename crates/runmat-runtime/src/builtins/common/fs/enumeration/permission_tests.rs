use super::*;
use runmat_filesystem::{DirEntry, FileHandle, FsProvider, OpenFlags};

struct DeniedDirectory {
    inner: MemoryFsProvider,
    deny_metadata: bool,
}
#[async_trait::async_trait(?Send)]
impl FsProvider for DeniedDirectory {
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/workspace"))
    }
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.inner.open(path, flags)
    }
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        if self.deny_metadata {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "denied metadata",
            ));
        }
        self.inner.metadata(path).await
    }
    async fn read_dir(&self, _path: &Path) -> io::Result<Vec<DirEntry>> {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "denied enumeration",
        ))
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.inner.read(path).await
    }
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.inner.write(path, data).await
    }
    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_file(path).await
    }
    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.inner.symlink_metadata(path).await
    }
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.inner.canonicalize(path).await
    }
    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.inner.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.inner.create_dir_all(path).await
    }
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_dir(path).await
    }
    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_dir_all(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.inner.rename(from, to).await
    }
    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.inner.set_readonly(path, readonly).await
    }
}
#[test]
fn wildcard_permission_errors_are_not_empty_matches() {
    let _lock = runmat_filesystem::provider_override_lock();
    for deny_metadata in [false, true] {
        let inner = MemoryFsProvider::with_current_dir("/workspace");
        inner
            .write_project_path("/workspace/source/file.m", b"")
            .unwrap();
        runmat_filesystem::with_provider_override(
            Arc::new(DeniedDirectory {
                inner,
                deny_metadata,
            }),
            || {
                for pattern in ["source/*.m", "source/**/*.m"] {
                    let error = futures::executor::block_on(matching(Path::new(pattern)))
                        .err()
                        .expect("permission failure");
                    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
                }
            },
        );
    }
}
