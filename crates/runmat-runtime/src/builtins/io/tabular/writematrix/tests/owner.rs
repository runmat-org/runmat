use super::*;
use async_trait::async_trait;
use runmat_filesystem::{
    DirEntry, FileHandle, FsFilePayloadContract, FsMetadata, FsProvider, FsResourceContract,
    FsResourceLease, FsResourceRevision, MemoryFsProvider, OpenFlags,
};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

struct RevocableLease(Arc<AtomicBool>);

impl FsResourceLease for RevocableLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("writematrix-test", b"v1"),
            opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
            opened_output_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
        }
    }

    fn is_valid(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

struct RevokeAfterFirstWrite {
    inner: Box<dyn FileHandle>,
    valid: Arc<AtomicBool>,
    writes: Arc<AtomicUsize>,
    interrupt_once: Arc<AtomicBool>,
    revoke_on_interrupt: bool,
    revoke_after_success: bool,
}

impl Read for RevokeAfterFirstWrite {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.inner.read(output)
    }
}

impl Write for RevokeAfterFirstWrite {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.interrupt_once.swap(false, Ordering::SeqCst) {
            if self.revoke_on_interrupt {
                self.valid.store(false, Ordering::SeqCst);
            }
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "fixture interrupt",
            ));
        }
        let result = self.inner.write(&bytes[..bytes.len().min(1)])?;
        self.writes.fetch_add(1, Ordering::SeqCst);
        if self.revoke_after_success {
            self.valid.store(false, Ordering::SeqCst);
        }
        Ok(result)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for RevokeAfterFirstWrite {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}

#[async_trait(?Send)]
impl FileHandle for RevokeAfterFirstWrite {}

struct RevokingProvider {
    memory: MemoryFsProvider,
    valid: Arc<AtomicBool>,
    writes: Arc<AtomicUsize>,
    interrupt_once: Arc<AtomicBool>,
    revoke_on_interrupt: bool,
    revoke_after_success: bool,
}

#[async_trait(?Send)]
impl FsProvider for RevokingProvider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(Arc::new(RevocableLease(self.valid.clone())))
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/tabular"))
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let inner = self.memory.open(path, flags)?;
        if flags.write {
            Ok(Box::new(RevokeAfterFirstWrite {
                inner,
                valid: self.valid.clone(),
                writes: self.writes.clone(),
                interrupt_once: self.interrupt_once.clone(),
                revoke_on_interrupt: self.revoke_on_interrupt,
                revoke_after_success: self.revoke_after_success,
            }))
        } else {
            Ok(inner)
        }
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.memory.read(path).await
    }
    async fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.memory.write(path, bytes).await
    }
    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_file(path).await
    }
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.metadata(path).await
    }
    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.symlink_metadata(path).await
    }
    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.memory.read_dir(path).await
    }
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.memory.canonicalize(path).await
    }
    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir_all(path).await
    }
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir(path).await
    }
    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir_all(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.memory.rename(from, to).await
    }
    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.memory.set_readonly(path, readonly).await
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn revoked_owner_after_partial_first_write_prevents_retry_and_success() {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir("/tabular");
    let writes = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(RevokingProvider {
        memory,
        valid: Arc::new(AtomicBool::new(true)),
        writes: writes.clone(),
        interrupt_once: Arc::new(AtomicBool::new(false)),
        revoke_on_interrupt: false,
        revoke_after_success: true,
    });
    let _guard = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir("/tabular").expect("cwd");
    let matrix = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("matrix");
    let error = block_on(super::super::writematrix_builtin(
        None,
        Value::Tensor(matrix),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("revoked.csv")]),
    ))
    .expect_err("revoked owner");
    assert!(error.message().contains("authority was revoked"));
    assert_eq!(writes.load(Ordering::SeqCst), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn interrupted_then_partial_writes_preserve_output() {
    let _lock = runmat_filesystem::provider_override_lock();
    let writes = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(RevokingProvider {
        memory: MemoryFsProvider::with_current_dir("/tabular"),
        valid: Arc::new(AtomicBool::new(true)),
        writes: writes.clone(),
        interrupt_once: Arc::new(AtomicBool::new(true)),
        revoke_on_interrupt: false,
        revoke_after_success: false,
    });
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir("/tabular").expect("cwd");
    let matrix = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("matrix");
    block_on(super::super::writematrix_builtin(
        None,
        Value::Tensor(matrix),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
            "interrupted.csv",
        )]),
    ))
    .expect("interrupted write retries");
    assert_eq!(
        provider
            .memory
            .read_project_path("/tabular/interrupted.csv")
            .expect("written bytes"),
        b"1,2\n"
    );
    assert!(writes.load(Ordering::SeqCst) > 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn revoked_owner_on_interrupted_write_prevents_retry() {
    let _lock = runmat_filesystem::provider_override_lock();
    let writes = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(RevokingProvider {
        memory: MemoryFsProvider::with_current_dir("/tabular"),
        valid: Arc::new(AtomicBool::new(true)),
        writes: writes.clone(),
        interrupt_once: Arc::new(AtomicBool::new(true)),
        revoke_on_interrupt: true,
        revoke_after_success: false,
    });
    let _guard = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir("/tabular").expect("cwd");
    let error = block_on(super::super::writematrix_builtin(
        None,
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
            "revoked-interrupt.csv",
        )]),
    ))
    .expect_err("revocation wins before retry");
    assert!(error.message().contains("authority was revoked"));
    assert_eq!(writes.load(Ordering::SeqCst), 0);
}
