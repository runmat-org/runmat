use super::*;
use futures::executor::block_on;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags, OpenOptions};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

struct FailingSeekHandle(Cursor<Vec<u8>>);

impl Read for FailingSeekHandle {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.0.read(buffer)
    }
}

impl Write for FailingSeekHandle {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for FailingSeekHandle {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        if matches!(position, SeekFrom::Start(0)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "restoring cursor failed",
            ));
        }
        self.0.seek(position)
    }
}

#[async_trait::async_trait(?Send)]
impl FileHandle for FailingSeekHandle {}

struct FailingSeekProvider;

fn unsupported() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "not used by this fixture")
}

#[async_trait::async_trait(?Send)]
impl FsProvider for FailingSeekProvider {
    fn open(&self, _path: &Path, _flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Ok(Box::new(FailingSeekHandle(Cursor::new(b"abc".to_vec()))))
    }
    async fn read(&self, _: &Path) -> io::Result<Vec<u8>> {
        Err(unsupported())
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unsupported())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unsupported())
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unsupported())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unsupported())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unsupported())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(unsupported())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn failed_cursor_rollback_is_sticky_and_retains_source_until_explicit_close() {
    let _registry_guard = test_guard();
    reset_for_tests();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let _provider = runmat_filesystem::replace_provider(Arc::new(FailingSeekProvider));
    let mut options = OpenOptions::new();
    options.read(true);
    let path = PathBuf::from("/rollback-failure.txt");
    let file = block_on(options.open_async(&path)).expect("open exact owner");
    let handle = Arc::new(StdMutex::new(Some(file)));
    let fid = register_file(RegisteredFile {
        path,
        permission: "r".to_string(),
        access: FileAccess::READ,
        machinefmt: "native".to_string(),
        encoding: "UTF-8".to_string(),
        handle: handle.clone(),
    });
    let mut operation = acquire_file(fid).expect("checkout");
    operation.restore_cursor_on_drop(0);
    std::io::Read::read_exact(operation.file_mut(), &mut [0; 1]).expect("read");
    drop(operation);
    let error = acquire_file(fid).err().expect("restore failure sticks");
    let FileOperationError::RestoreFailed(source) = error else {
        panic!("restore failure cannot masquerade as closed or busy");
    };
    assert_eq!(source.kind(), io::ErrorKind::InvalidData);
    assert_eq!(source.to_string(), "restoring cursor failed");
    let wrapped = FileOperationError::RestoreFailed(source).into_io_error();
    assert_eq!(wrapped.kind(), io::ErrorKind::Other);
    assert_eq!(
        wrapped
            .get_ref()
            .and_then(|error| error.source())
            .map(ToString::to_string),
        Some("restoring cursor failed".to_string())
    );
    assert!(Arc::ptr_eq(
        &handle,
        &snapshot_file(fid).expect("same entry").handle
    ));
    block_on(close_async(fid))
        .expect("close failed entry")
        .expect("entry");
    reset_for_tests();
}
