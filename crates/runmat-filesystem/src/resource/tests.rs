use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_direct_source_excludes_os_backing_but_memory_retains_managed_backing() {
    let native = crate::NativeFsProvider
        .managed_read_contract()
        .expect("native direct contract");
    assert_eq!(
        native.backing,
        FsManagedReadBacking::ExternalToRustManagedHeap
    );
    let memory = crate::MemoryFsProvider::new()
        .managed_read_contract()
        .expect("MemoryFS direct contract");
    assert_eq!(
        memory.backing,
        FsManagedReadBacking::RetainedManagedPayloadUnaccounted
    );
}
#[cfg(not(target_arch = "wasm32"))]
use crate::NativeFsProvider;
use crate::{FsMetadata, MemoryFsProvider};
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_opened_extent_bounds_growth_and_keeps_small_range_reads_small() {
    use std::fs::{self, OpenOptions};
    use std::io::Write;

    let path = std::env::temp_dir().join(format!("runmat-bounded-input-{}", uuid::Uuid::new_v4()));
    let mut writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("isolated input");
    writer.write_all(b"abcd").expect("initial bytes");
    writer.flush().expect("initial flush");
    let provider = NativeFsProvider;
    let mut opened = futures::executor::block_on(provider.open_bounded_input(&path))
        .expect("native open")
        .expect("direct opened source");
    assert_eq!(opened.opened_extent(), 4);
    assert!(opened.is_valid());
    writer.write_all(b"efgh").expect("growth after open");
    writer.flush().expect("growth flush");
    let mut bytes = Vec::new();
    opened.read_to_end(&mut bytes).expect("bounded read");
    assert_eq!(bytes, b"abcd");
    assert_eq!(opened.seek(SeekFrom::End(0)).expect("bounded end"), 4);
    assert!(opened.seek(SeekFrom::Start(5)).is_err());
    assert_eq!(
        futures::executor::block_on(opened.metadata_async())
            .expect("opened metadata")
            .len(),
        4
    );
    drop(opened);
    drop(writer);
    fs::remove_file(path).expect("remove isolated input");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_managed_read_contract_matches_retained_open_authority() {
    use std::fs;

    let provider: Arc<dyn FsProvider> = Arc::new(NativeFsProvider);
    let contract = provider
        .managed_read_contract()
        .expect("native direct handle has a typed managed request bound");
    assert_eq!(contract.capture_requested_heap_bytes, 0);
    assert_eq!(
        contract.opened_requested_heap_bytes,
        contract.capture_requested_heap_bytes + std::mem::size_of::<fs::File>() as u64
    );
    let captured = capture_resource_lease_for_effect(provider.clone())
        .expect("capture")
        .expect("native lease");
    assert_eq!(captured.contract().revision, contract.revision);
    assert_eq!(
        captured
            .managed_opened_input_cost()
            .expect("lease agrees with provider's pure claim")
            .requested_heap_bytes,
        contract.opened_requested_heap_bytes
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn sparse_large_native_source_needs_only_a_small_caller_buffer() {
    use std::fs::{self, OpenOptions};

    let path = std::env::temp_dir().join(format!("runmat-bounded-large-{}", uuid::Uuid::new_v4()));
    let writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("isolated sparse input");
    writer.set_len(1_000_000_000).expect("sparse extent");
    let provider = NativeFsProvider;
    let mut opened = futures::executor::block_on(provider.open_bounded_input(&path))
        .expect("native open")
        .expect("direct opened source");
    assert_eq!(opened.opened_extent(), 1_000_000_000);
    opened.seek(SeekFrom::Start(999_999_996)).expect("tail");
    let mut tail = [1u8; 8];
    assert_eq!(opened.read(&mut tail).expect("tail read"), 4);
    assert_eq!(&tail[..4], &[0; 4]);
    assert_eq!(opened.read(&mut tail).expect("bounded eof"), 0);
    drop(opened);
    drop(writer);
    fs::remove_file(path).expect("remove isolated sparse input");
}

#[cfg(unix)]
#[test]
fn path_replacement_does_not_redirect_the_opened_source() {
    use std::fs;

    let root = std::env::temp_dir().join(format!("runmat-bounded-rename-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).expect("isolated root");
    let path = root.join("source.wav");
    let old_path = root.join("old.wav");
    fs::write(&path, b"old!").expect("original source");
    let provider = NativeFsProvider;
    let mut opened = futures::executor::block_on(provider.open_bounded_input(&path))
        .expect("native open")
        .expect("direct opened source");
    fs::rename(&path, &old_path).expect("move original after open");
    fs::write(&path, b"new-and-larger").expect("replace pathname");
    let mut bytes = Vec::new();
    opened
        .read_to_end(&mut bytes)
        .expect("retained source read");
    assert_eq!(bytes, b"old!");
    assert_eq!(opened.opened_extent(), 4);
    drop(opened);
    fs::remove_file(path).expect("remove replacement");
    fs::remove_file(old_path).expect("remove original");
    fs::remove_dir(root).expect("remove root");
}

#[test]
fn provider_revisions_are_domain_separated_and_extensible() {
    let native = FsResourceRevision::for_provider("runmat.native", b"direct-files-v1");
    assert_ne!(
        native,
        FsResourceRevision::for_provider("third-party.provider", b"direct-files-v1")
    );
    assert_ne!(
        native,
        FsResourceRevision::for_provider("runmat.native", b"direct-files-v2")
    );
    assert_ne!(
        FsResourceRevision::for_provider("ab", b"c"),
        FsResourceRevision::for_provider("a", b"bc")
    );
    assert_eq!(
        serde_json::to_value(native).expect("typed revision serializes"),
        serde_json::Value::String(native.to_string())
    );
}

#[test]
fn memory_provider_has_its_own_immutable_snapshot_contract() {
    let provider: Arc<dyn FsProvider> = Arc::new(MemoryFsProvider::new());
    assert!(provider.resource_lease().is_none());
    let contract = provider
        .managed_read_contract()
        .expect("memory direct snapshot");
    assert_eq!(contract.capture_requested_heap_bytes, 0);
    assert!(contract.opened_requested_heap_bytes > 0);
    let lease = capture_resource_lease_for_effect(provider)
        .expect("capture")
        .expect("memory snapshot lease");
    assert_eq!(lease.contract().revision, contract.revision);
    assert_eq!(lease.contract().opened_input_handle, None);
    assert_eq!(
        lease
            .managed_opened_input_cost()
            .unwrap()
            .requested_heap_bytes,
        contract.opened_requested_heap_bytes
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_claim_retains_exact_provider_without_claiming_codec_memory() {
    let provider: Arc<dyn FsProvider> = Arc::new(NativeFsProvider);
    let captured = capture_resource_lease(provider.clone()).expect("native direct handles");
    assert!(Arc::ptr_eq(captured.provider(), &provider));
    drop(provider); // The exact provider remains alive with the lease.
    assert!(captured.is_valid());
    assert_eq!(
        captured.contract(),
        FsResourceContract {
            revision: FsResourceRevision::for_provider(
                "runmat.filesystem.native",
                b"direct-file-handles-bounded-open-managed-v3",
            ),
            opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
            opened_output_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
        }
    );
}

struct RevocableLease(Arc<AtomicBool>);

impl FsResourceLease for RevocableLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("test.revocable", b"direct-file-handles-v1"),
            opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
            opened_output_handle: None,
        }
    }

    fn is_valid(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[test]
fn captured_lease_detects_revocation_after_suspension() {
    let valid = Arc::new(AtomicBool::new(true));
    let provider: Arc<dyn FsProvider> = Arc::new(MemoryFsProvider::new());
    let lease: Arc<dyn FsResourceLease> = Arc::new(RevocableLease(valid.clone()));
    let captured = CapturedFsResourceLease {
        provider,
        contract: lease.contract(),
        lease: FsResourceLeaseHandle::Owned(lease),
    };
    assert!(captured.is_valid());
    valid.store(false, Ordering::Release);
    assert!(!captured.is_valid());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn bounded_opened_read_rechecks_owner_before_each_effect() {
    use std::fs;

    let path = std::env::temp_dir().join(format!("runmat-revoked-input-{}", uuid::Uuid::new_v4()));
    fs::write(&path, b"sound").expect("isolated input");
    let file = fs::File::open(&path).expect("opened input");
    let metadata = FsMetadata::from(file.metadata().expect("opened metadata"));
    let valid = Arc::new(AtomicBool::new(true));
    let lease: Arc<dyn FsResourceLease> = Arc::new(RevocableLease(valid.clone()));
    let mut opened =
        FsBoundedInput::new(Box::new(file), metadata, lease).expect("direct opened input");
    let mut first = [0u8; 1];
    assert_eq!(opened.read(&mut first).expect("first read"), 1);
    valid.store(false, Ordering::Release);
    assert_eq!(
        opened.read(&mut first).unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
    assert_eq!(
        opened.seek(SeekFrom::Start(0)).unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
    drop(opened);
    fs::remove_file(path).expect("remove isolated input");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn opened_handle_rechecks_revision_after_seek_and_read() {
    use std::io::Write;
    use std::sync::atomic::AtomicUsize;

    struct ChangingRevision(Arc<AtomicUsize>);
    impl FsResourceLease for ChangingRevision {
        fn contract(&self) -> FsResourceContract {
            FsResourceContract {
                revision: FsResourceRevision::for_provider(
                    "test.opened-operation",
                    &[self.0.load(Ordering::SeqCst) as u8],
                ),
                opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
                opened_output_handle: None,
            }
        }
        fn is_valid(&self) -> bool {
            true
        }
    }

    struct ChangingHandle {
        cursor: std::io::Cursor<Vec<u8>>,
        revision: Arc<AtomicUsize>,
        change_on_seek: bool,
        change_on_read: bool,
    }
    impl Read for ChangingHandle {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let read = self.cursor.read(buffer)?;
            if self.change_on_read {
                self.revision.fetch_add(1, Ordering::SeqCst);
            }
            Ok(read)
        }
    }
    impl Write for ChangingHandle {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.cursor.write(buffer)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.cursor.flush()
        }
    }
    impl Seek for ChangingHandle {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            let offset = self.cursor.seek(position)?;
            if self.change_on_seek {
                self.revision.fetch_add(1, Ordering::SeqCst);
            }
            Ok(offset)
        }
    }
    #[async_trait::async_trait(?Send)]
    impl crate::FileHandle for ChangingHandle {}

    let metadata = FsMetadata {
        file_type: crate::FsFileType::File,
        len: 4,
        modified: None,
        readonly: false,
        hash: None,
    };
    let revision = Arc::new(AtomicUsize::new(0));
    let lease: Arc<dyn FsResourceLease> = Arc::new(ChangingRevision(revision.clone()));
    let handle = ChangingHandle {
        cursor: std::io::Cursor::new(b"abcd".to_vec()),
        revision: revision.clone(),
        change_on_seek: true,
        change_on_read: false,
    };
    let error = FsBoundedInput::new(Box::new(handle), metadata.clone(), lease.clone())
        .err()
        .expect("revision changed in initial seek");
    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);

    revision.store(0, Ordering::SeqCst);
    let handle = ChangingHandle {
        cursor: std::io::Cursor::new(b"abcd".to_vec()),
        revision: revision.clone(),
        change_on_seek: false,
        change_on_read: true,
    };
    let mut input = FsBoundedInput::new(Box::new(handle), metadata, lease).unwrap();
    let mut byte = [0; 1];
    let error = input.read(&mut byte).expect_err("revision changed in read");
    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
}

#[test]
fn bounded_input_rejects_inconsistent_handle_positions_and_counts() {
    use std::io::{Cursor, Write};

    struct MisreportingHandle {
        bytes: Cursor<Vec<u8>>,
        wrong_seek: Arc<AtomicBool>,
        wrong_count: bool,
    }
    impl Read for MisreportingHandle {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let actual = self.bytes.read(buffer)?;
            Ok(actual + usize::from(self.wrong_count))
        }
    }
    impl Write for MisreportingHandle {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.bytes.write(buffer)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.bytes.flush()
        }
    }
    impl Seek for MisreportingHandle {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            let actual = self.bytes.seek(position)?;
            Ok(actual + u64::from(self.wrong_seek.load(Ordering::SeqCst)))
        }
    }
    #[async_trait::async_trait(?Send)]
    impl crate::FileHandle for MisreportingHandle {}

    let metadata = FsMetadata {
        file_type: crate::FsFileType::File,
        len: 4,
        modified: None,
        readonly: false,
        hash: None,
    };
    let lease: Arc<dyn FsResourceLease> = Arc::new(RevocableLease(Arc::new(AtomicBool::new(true))));
    let wrong_seek = Arc::new(AtomicBool::new(true));
    let make_handle = |wrong_count| MisreportingHandle {
        bytes: Cursor::new(b"abcd".to_vec()),
        wrong_seek: wrong_seek.clone(),
        wrong_count,
    };
    assert_eq!(
        FsBoundedInput::new(
            Box::new(make_handle(false)),
            metadata.clone(),
            lease.clone()
        )
        .err()
        .expect("wrong initial position")
        .kind(),
        io::ErrorKind::InvalidData,
    );

    wrong_seek.store(false, Ordering::SeqCst);
    let mut input = FsBoundedInput::new(
        Box::new(make_handle(false)),
        metadata.clone(),
        lease.clone(),
    )
    .unwrap();
    wrong_seek.store(true, Ordering::SeqCst);
    assert_eq!(
        input.seek(SeekFrom::Start(2)).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );

    wrong_seek.store(false, Ordering::SeqCst);
    let mut input = FsBoundedInput::new(Box::new(make_handle(true)), metadata, lease).unwrap();
    let mut byte = [0; 1];
    assert_eq!(
        input.read(&mut byte).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}
