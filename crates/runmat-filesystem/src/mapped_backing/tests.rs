use super::*;
use crate::{
    DirEntry, FileHandle, FsFilePayloadContract, FsMetadata, FsProvider, FsResourceContract,
    FsResourceLease, FsResourceRevision, MemoryFsProvider, OpenFlags,
};
use async_trait::async_trait;
use futures::executor::block_on;
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

struct RevokedLease;

impl FsResourceLease for RevokedLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("test.mapped", b"revoked"),
            opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
            opened_output_handle: None,
        }
    }

    fn is_valid(&self) -> bool {
        false
    }
}

struct CountedOpenProvider(AtomicUsize);

#[async_trait(?Send)]
impl FsProvider for CountedOpenProvider {
    async fn open_mapped_backing(
        &self,
        _: &Path,
        _: bool,
    ) -> io::Result<Option<Box<dyn FsMappedBacking>>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    }
    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(Arc::new(RevokedLease))
    }
    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        unreachable!()
    }
    async fn read(&self, _: &Path) -> io::Result<Vec<u8>> {
        unreachable!()
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        unreachable!()
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        unreachable!()
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        unreachable!()
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        unreachable!()
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        unreachable!()
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        unreachable!()
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        unreachable!()
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        unreachable!()
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        unreachable!()
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        unreachable!()
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        unreachable!()
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        unreachable!()
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_path_or_revoked_lease_does_not_call_provider_open() {
    let provider = Arc::new(CountedOpenProvider(AtomicUsize::new(0)));
    let erased: Arc<dyn FsProvider> = provider.clone();
    let relative = block_on(capture_mapped_backing_on(
        erased.clone(),
        PathBuf::from("relative"),
        false,
    ))
    .err()
    .expect("relative path rejected");
    assert_eq!(relative.kind(), ErrorKind::InvalidInput);
    #[cfg(target_arch = "wasm32")]
    let absolute = PathBuf::from("/mapped-file");
    #[cfg(not(target_arch = "wasm32"))]
    let absolute = std::env::temp_dir().join("mapped-file");
    let revoked = block_on(capture_mapped_backing_on(erased, absolute, false))
        .err()
        .expect("revoked lease rejected");
    assert_eq!(revoked.kind(), ErrorKind::PermissionDenied);
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
}

struct RevokedAfterWrite {
    current: Arc<AtomicBool>,
    committed: Arc<Mutex<Vec<u8>>>,
    fabricated_committed_count: Option<usize>,
}

#[async_trait(?Send)]
impl FsMappedBacking for RevokedAfterWrite {
    fn opened_extent(&self) -> u64 {
        4
    }

    fn writable(&self) -> bool {
        true
    }

    fn check_current(&self) -> io::Result<()> {
        if self.current.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(io::Error::new(ErrorKind::PermissionDenied, "owner revoked"))
        }
    }

    async fn read_at(&mut self, _offset: u64, _buffer: &mut [u8]) -> io::Result<usize> {
        unreachable!("write-only fixture")
    }

    async fn write_at(
        &mut self,
        _offset: u64,
        bytes: &[u8],
    ) -> Result<FsMappedWriteOutcome, FsMappedWriteFailure> {
        if let Some(count) = self.fabricated_committed_count {
            return Err(FsMappedWriteFailure::committed(
                count,
                io::Error::new(ErrorKind::BrokenPipe, "provider failure cause"),
            ));
        }
        let accepted = bytes.len().min(2);
        self.committed
            .lock()
            .unwrap()
            .extend_from_slice(&bytes[..accepted]);
        self.current.store(false, Ordering::SeqCst);
        Ok(FsMappedWriteOutcome {
            accepted_bytes: accepted,
            requested_bytes: bytes.len(),
        })
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn post_write_revocation_reports_committed_partial_bytes_and_cause() {
    let current = Arc::new(AtomicBool::new(true));
    let committed = Arc::new(Mutex::new(Vec::new()));
    let mut backing = CapturedMappedBacking {
        provider: Arc::new(MemoryFsProvider::new()),
        path: PathBuf::from("/fixture"),
        resource: None,
        backing: Box::new(RevokedAfterWrite {
            current: current.clone(),
            committed: committed.clone(),
            fabricated_committed_count: None,
        }),
    };
    let failure = block_on(backing.write_at(0, b"abcd")).unwrap_err();
    assert_eq!(
        failure.disposition(),
        FsMappedWriteDisposition::Committed { accepted_bytes: 2 }
    );
    assert_eq!(failure.io_error().kind(), ErrorKind::PermissionDenied);
    assert_eq!(&*committed.lock().unwrap(), b"ab");
    assert!(!current.load(Ordering::SeqCst));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn impossible_provider_committed_count_retains_cause_as_indeterminate() {
    let mut backing = CapturedMappedBacking {
        provider: Arc::new(MemoryFsProvider::new()),
        path: PathBuf::from("/fixture"),
        resource: None,
        backing: Box::new(RevokedAfterWrite {
            current: Arc::new(AtomicBool::new(true)),
            committed: Arc::new(Mutex::new(Vec::new())),
            fabricated_committed_count: Some(5),
        }),
    };
    let failure = block_on(backing.write_at(0, b"abcd")).unwrap_err();
    assert_eq!(
        failure.disposition(),
        FsMappedWriteDisposition::Indeterminate { attempted_bytes: 4 }
    );
    assert_eq!(failure.io_error().kind(), ErrorKind::BrokenPipe);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn memory_backing_tracks_live_in_place_bytes_and_rejects_replacement() {
    let _lock = crate::provider_override_lock();
    let first = Arc::new(MemoryFsProvider::with_current_dir("/mapped"));
    first
        .write_project_path("/mapped/data.bin", b"abcdef")
        .unwrap();
    let _selected = crate::replace_provider(first.clone());
    crate::set_current_dir("/mapped").unwrap();
    let mut backing = block_on(capture_mapped_backing("data.bin", true))
        .unwrap()
        .expect("memory provider supports opened ranges");
    assert_eq!(backing.path(), Path::new("/mapped/data.bin"));
    assert_eq!(backing.opened_extent(), 6);

    let second = Arc::new(MemoryFsProvider::with_current_dir("/other"));
    second
        .write_project_path("/other/data.bin", b"wrong!")
        .unwrap();
    let _replacement = crate::replace_provider(second.clone());
    crate::set_current_dir("/other").unwrap();
    assert_eq!(
        block_on(backing.write_at(2, b"XY")).unwrap(),
        FsMappedWriteOutcome {
            accepted_bytes: 2,
            requested_bytes: 2
        }
    );
    assert_eq!(
        first.read_project_path("/mapped/data.bin").unwrap(),
        b"abXYef"
    );
    assert_eq!(
        second.read_project_path("/other/data.bin").unwrap(),
        b"wrong!"
    );

    first
        .write_project_path("/mapped/data.bin", b"123456")
        .unwrap();
    let mut read = [0; 3];
    assert_eq!(block_on(backing.read_at(1, &mut read)).unwrap(), 3);
    assert_eq!(&read, b"234");
    assert_eq!(
        block_on(backing.write_at(5, b"too long"))
            .unwrap_err()
            .io_error()
            .kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        first.read_project_path("/mapped/data.bin").unwrap(),
        b"123456"
    );

    block_on(first.remove_file(Path::new("/mapped/data.bin"))).unwrap();
    first
        .write_project_path("/mapped/data.bin", b"reused")
        .unwrap();
    assert_eq!(
        backing.check_current().unwrap_err().kind(),
        ErrorKind::PermissionDenied
    );
    assert_eq!(
        block_on(backing.write_at(0, b"X"))
            .unwrap_err()
            .io_error()
            .kind(),
        ErrorKind::PermissionDenied
    );
    assert_eq!(
        first.read_project_path("/mapped/data.bin").unwrap(),
        b"reused"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn memory_readonly_and_truncation_do_not_extend_opened_backing() {
    let provider = MemoryFsProvider::with_current_dir("/mapped");
    provider
        .write_project_path("/mapped/data.bin", b"abcd")
        .unwrap();
    let mut backing = block_on(provider.open_mapped_backing(Path::new("/mapped/data.bin"), false))
        .unwrap()
        .expect("memory range backing");
    assert!(!backing.writable());
    assert_eq!(
        block_on(backing.write_at(0, b"X"))
            .unwrap_err()
            .io_error()
            .kind(),
        ErrorKind::PermissionDenied
    );
    provider
        .write_project_path("/mapped/data.bin", b"ab")
        .unwrap();
    let mut read = [0; 4];
    assert_eq!(block_on(backing.read_at(0, &mut read)).unwrap(), 2);
    assert_eq!(&read[..2], b"ab");

    let mut writer = block_on(provider.open_mapped_backing(Path::new("/mapped/data.bin"), true))
        .unwrap()
        .expect("memory writable range backing");
    provider
        .write_project_path("/mapped/data.bin", b"a")
        .unwrap();
    assert_eq!(
        block_on(writer.write_at(1, b"Z"))
            .unwrap_err()
            .io_error()
            .kind(),
        ErrorKind::UnexpectedEof
    );
    assert_eq!(
        provider.read_project_path("/mapped/data.bin").unwrap(),
        b"a"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn old_memory_snapshot_handle_cannot_overwrite_mapped_write_or_replacement() {
    let provider = MemoryFsProvider::with_current_dir("/mapped");
    let path = Path::new("/mapped/data.bin");
    provider.write_project_path(path, b"abcdef").unwrap();
    let flags = OpenFlags {
        read: true,
        write: true,
        ..OpenFlags::default()
    };
    let mut old = provider.open(path, &flags).unwrap();
    old.write_all(b"stale!").unwrap();
    let mut mapped = block_on(provider.open_mapped_backing(path, true))
        .unwrap()
        .expect("memory mapped owner");
    assert_eq!(
        block_on(mapped.write_at(0, b"Q")).unwrap().accepted_bytes,
        1
    );
    assert_eq!(old.flush().unwrap_err().kind(), ErrorKind::PermissionDenied);
    drop(old);
    assert_eq!(provider.read_project_path(path).unwrap(), b"Qbcdef");

    let mut old = provider.open(path, &flags).unwrap();
    old.write_all(b"stale!").unwrap();
    block_on(provider.remove_file(path)).unwrap();
    provider.write_project_path(path, b"newnew").unwrap();
    assert_eq!(old.flush().unwrap_err().kind(), ErrorKind::PermissionDenied);
    drop(old);
    assert_eq!(provider.read_project_path(path).unwrap(), b"newnew");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_backing_writes_only_inside_its_opened_extent() {
    use crate::NativeFsProvider;
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("data.bin");
    std::fs::write(&path, b"abcd").unwrap();
    let provider = NativeFsProvider;
    let mut backing = block_on(provider.open_mapped_backing(&path, true))
        .unwrap()
        .expect("native opened range backing");
    assert_eq!(backing.opened_extent(), 4);
    assert_eq!(
        block_on(backing.write_at(1, b"XY")).unwrap(),
        FsMappedWriteOutcome {
            accepted_bytes: 2,
            requested_bytes: 2
        }
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"aXYd");
    assert_eq!(
        block_on(backing.write_at(4, b"Z"))
            .unwrap_err()
            .io_error()
            .kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 4);
}

#[cfg(unix)]
#[test]
fn native_backing_pins_opened_file_and_exposes_external_in_place_edits() {
    use crate::NativeFsProvider;
    let root = tempfile::tempdir().unwrap();
    let original = root.path().join("data.bin");
    let moved = root.path().join("moved.bin");
    std::fs::write(&original, b"abcdef").unwrap();
    let provider = NativeFsProvider;
    let mut backing = block_on(provider.open_mapped_backing(&original, true))
        .unwrap()
        .expect("native opened range backing");
    std::fs::rename(&original, &moved).unwrap();
    std::fs::write(&original, b"wrong!").unwrap();
    assert_eq!(
        block_on(backing.write_at(2, b"XY")).unwrap(),
        FsMappedWriteOutcome {
            accepted_bytes: 2,
            requested_bytes: 2
        }
    );
    assert_eq!(std::fs::read(&moved).unwrap(), b"abXYef");
    assert_eq!(std::fs::read(&original).unwrap(), b"wrong!");
    std::fs::write(&moved, b"123456").unwrap();
    let mut read = [0; 3];
    assert_eq!(block_on(backing.read_at(1, &mut read)).unwrap(), 3);
    assert_eq!(&read, b"234");
    assert_eq!(
        block_on(backing.write_at(6, b"X"))
            .unwrap_err()
            .io_error()
            .kind(),
        ErrorKind::InvalidInput
    );
    std::fs::write(&moved, b"a").unwrap();
    assert_eq!(
        block_on(backing.write_at(0, b"X"))
            .unwrap_err()
            .io_error()
            .kind(),
        ErrorKind::UnexpectedEof
    );
}
