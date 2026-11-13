use super::*;
use async_trait::async_trait;
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, FsResourceContract, FsResourceLease,
    FsResourceRevision, OpenFlags,
};
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

struct Lease(Arc<AtomicBool>);

impl FsResourceLease for Lease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("xlswrite.revocation.fixture", b"v1"),
            opened_input_handle: None,
            opened_output_handle: None,
        }
    }

    fn is_valid(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

struct RevocableProvider {
    memory: MemoryFsProvider,
    valid: Arc<AtomicBool>,
}

#[async_trait(?Send)]
impl FsProvider for RevocableProvider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        self.memory.current_dir_override()
    }

    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(Arc::new(Lease(self.valid.clone())))
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.memory.open(path, flags)
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.memory.read(path).await
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.memory.write(path, data).await
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
fn revoked_lease_error_keeps_retained_workbook_valid() {
    let _lock = runmat_filesystem::provider_override_lock();
    let valid = Arc::new(AtomicBool::new(true));
    let provider = Arc::new(RevocableProvider {
        memory: MemoryFsProvider::with_current_dir("/xlswrite-overlay"),
        valid: valid.clone(),
    });
    let _restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new("/xlswrite-overlay/out.xlsx")))
        .expect("captured owner")
        .path("/xlswrite-overlay/out.xlsx".into())
        .expect("selected path");
    let mut workbook = XlsWorkbook::new();
    futures::executor::block_on(seed_workbook(&mut workbook, &selected))
        .expect("initial valid sheet");
    let incoming = futures::executor::block_on(table("new", &selected));
    valid.store(false, Ordering::Release);
    let request = XlsWriteRequest {
        sheet: SheetSelector::Default,
        range: RangeStart { row: 1, col: 1 },
    };
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let error = futures::executor::block_on(workbook.overlay(
        &request,
        incoming,
        &mut cooperation,
        &selected,
    ))
    .expect_err("revoked selected owner");
    assert_eq!(error.identifier(), XLSWRITE_ERROR_IO.identifier);

    let replacement = Arc::new(MemoryFsProvider::with_current_dir("/xlswrite-overlay"));
    let _replacement = runmat_filesystem::replace_provider(replacement);
    let recovered = TabularFsOwner::capture(Some(Path::new("/xlswrite-overlay/out.xlsx")))
        .expect("new captured owner")
        .path("/xlswrite-overlay/out.xlsx".into())
        .expect("new selected path");
    futures::executor::block_on(recovered_workbook(&mut workbook, &recovered))
        .expect("retained workbook remains valid after lease failure");
}
