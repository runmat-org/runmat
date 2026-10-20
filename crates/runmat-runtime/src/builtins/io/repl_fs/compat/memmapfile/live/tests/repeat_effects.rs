use super::*;
use runmat_filesystem::{
    DirEntry, FileHandle, FsMappedBacking, FsMappedWriteFailure, FsMappedWriteOutcome, FsMetadata,
    FsProvider, OpenFlags,
};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct CountedProvider {
    root: PathBuf,
    bytes: Arc<Mutex<Vec<u8>>>,
    opens: Arc<AtomicUsize>,
    reads: Arc<AtomicUsize>,
    writes: Arc<AtomicUsize>,
    pending_reads: bool,
    backing_drops: Arc<AtomicUsize>,
}

struct CountedBacking {
    bytes: Arc<Mutex<Vec<u8>>>,
    reads: Arc<AtomicUsize>,
    writes: Arc<AtomicUsize>,
    pending_reads: bool,
    backing_drops: Arc<AtomicUsize>,
}

impl Drop for CountedBacking {
    fn drop(&mut self) {
        self.backing_drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait::async_trait(?Send)]
impl FsMappedBacking for CountedBacking {
    fn opened_extent(&self) -> u64 {
        self.bytes.lock().unwrap().len() as u64
    }

    fn writable(&self) -> bool {
        true
    }

    fn check_current(&self) -> io::Result<()> {
        Ok(())
    }

    async fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if self.pending_reads {
            std::future::pending::<()>().await;
        }
        let bytes = self.bytes.lock().unwrap();
        let start = usize::try_from(offset).unwrap();
        let count = buffer.len().min(bytes.len().saturating_sub(start));
        buffer[..count].copy_from_slice(&bytes[start..start + count]);
        Ok(count)
    }

    async fn write_at(
        &mut self,
        offset: u64,
        payload: &[u8],
    ) -> Result<FsMappedWriteOutcome, FsMappedWriteFailure> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        let mut bytes = self.bytes.lock().unwrap();
        let start = usize::try_from(offset).unwrap();
        bytes[start..start + payload.len()].copy_from_slice(payload);
        Ok(FsMappedWriteOutcome {
            accepted_bytes: payload.len(),
            requested_bytes: payload.len(),
        })
    }
}

#[async_trait::async_trait(?Send)]
impl FsProvider for CountedProvider {
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(self.root.clone())
    }

    async fn open_mapped_backing(
        &self,
        _path: &Path,
        _writable: bool,
    ) -> io::Result<Option<Box<dyn FsMappedBacking>>> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        Ok(Some(Box::new(CountedBacking {
            bytes: self.bytes.clone(),
            reads: self.reads.clone(),
            writes: self.writes.clone(),
            pending_reads: self.pending_reads,
            backing_drops: self.backing_drops.clone(),
        })))
    }

    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        unreachable!("mapped fixture does not use ordinary open")
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
fn invalid_repeat_forms_do_not_open_or_touch_mapped_bytes() {
    let _lock = vfs::provider_override_lock();
    #[cfg(target_arch = "wasm32")]
    let root = PathBuf::from("/repeat-counted");
    #[cfg(not(target_arch = "wasm32"))]
    let root = std::env::temp_dir().join("runmat-repeat-counted");
    let provider = Arc::new(CountedProvider {
        root,
        bytes: Arc::new(Mutex::new(vec![1, 2, 3, 4])),
        opens: Arc::new(AtomicUsize::new(0)),
        reads: Arc::new(AtomicUsize::new(0)),
        writes: Arc::new(AtomicUsize::new(0)),
        pending_reads: false,
        backing_drops: Arc::new(AtomicUsize::new(0)),
    });
    let _selected = vfs::replace_provider(provider.clone());
    let invalid = [
        Value::Num(f64::NEG_INFINITY),
        Value::Num(f64::NAN),
        Value::Num(1.5),
        Value::String("Inf".into()),
    ];
    for value in &invalid {
        let error = futures::executor::block_on(super::super::super::super::memmapfile_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("data.bin".into()),
                Value::String("Repeat".into()),
                value.clone(),
            ]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:memmapfile:InvalidControl"));
    }
    assert_eq!(provider.opens.load(Ordering::SeqCst), 0);
    assert_eq!(provider.reads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.writes.load(Ordering::SeqCst), 0);

    let mapped = futures::executor::block_on(super::super::super::super::memmapfile_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "data.bin".into(),
        )]),
    ))
    .unwrap();
    assert_eq!(provider.opens.load(Ordering::SeqCst), 1);
    for value in invalid {
        let error =
            futures::executor::block_on(subsasgn(None, mapped.clone(), property("Repeat"), value))
                .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:memmapfile:InvalidControl"));
        let Value::Object(object) = &mapped else {
            unreachable!()
        };
        assert_eq!(object.memmap_receiver().unwrap().repeat, None);
    }
    assert_eq!(provider.opens.load(Ordering::SeqCst), 1);
    assert_eq!(provider.reads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.writes.load(Ordering::SeqCst), 0);
    assert_eq!(*provider.bytes.lock().unwrap(), [1, 2, 3, 4]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abandoned_pending_mapped_read_releases_session_backing_without_replay() {
    use crate::execution::RuntimeExecutionService;
    use std::future::Future;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    let _lock = vfs::provider_override_lock();
    #[cfg(target_arch = "wasm32")]
    let root = PathBuf::from("/pending-mapped-read");
    #[cfg(not(target_arch = "wasm32"))]
    let root = std::env::temp_dir().join("runmat-pending-mapped-read");
    let provider = Arc::new(CountedProvider {
        root,
        bytes: Arc::new(Mutex::new(vec![1, 2, 3, 4])),
        opens: Arc::new(AtomicUsize::new(0)),
        reads: Arc::new(AtomicUsize::new(0)),
        writes: Arc::new(AtomicUsize::new(0)),
        pending_reads: true,
        backing_drops: Arc::new(AtomicUsize::new(0)),
    });
    let _selected = vfs::replace_provider(provider.clone());
    let runtime = crate::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let mapped = futures::executor::block_on(super::super::super::super::memmapfile_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "data.bin".into(),
        )]),
    ))
    .unwrap();
    let mut call = Box::pin(subsref(
        Some(runtime.clone()),
        mapped.clone(),
        property("Data"),
    ));
    let waker = futures::task::noop_waker_ref();
    let mut task = Context::from_waker(waker);
    assert!(matches!(call.as_mut().poll(&mut task), Poll::Pending));
    assert_eq!(provider.reads.load(Ordering::SeqCst), 1);
    drop(call);
    drop(runtime);
    assert_eq!(provider.backing_drops.load(Ordering::SeqCst), 1);
    assert_eq!(provider.writes.load(Ordering::SeqCst), 0);
    assert_eq!(*provider.bytes.lock().unwrap(), [1, 2, 3, 4]);
    drop(mapped);
    assert_eq!(provider.backing_drops.load(Ordering::SeqCst), 1);
}

struct NumericDownloadProvider {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    downloads: AtomicUsize,
}

impl runmat_accelerate_api::AccelProvider for NumericDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "memmapfile explicit control witness".into()
    }
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        anyhow::bail!("mapped method does not upload")
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Ok(runmat_accelerate_api::HostTensorOwned {
                data: vec![7.0],
                shape: vec![1, 1],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
        })
    }
    fn download_numeric<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Ok(runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::F64(vec![7.0]),
                shape: vec![1, 1],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
        })
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn explicit_gpu_mapped_rhs_is_gated_then_gathered_once_in_runmat_mode() {
    let _accel_lock = crate::builtins::common::test_support::accel_test_lock();
    let _fs_lock = vfs::provider_override_lock();
    let gpu = Box::leak(Box::new(NumericDownloadProvider {
        owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    unsafe { runmat_accelerate_api::register_provider(gpu) };
    let _gpu_selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(gpu));
    let allocation = gpu
        .owner
        .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
            data: runmat_accelerate_api::HostNumericDataOwned::F64(vec![7.0]),
            shape: vec![1, 1],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .unwrap();
    let explicit = allocation
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    #[cfg(target_arch = "wasm32")]
    let root = PathBuf::from("/explicit-mapped");
    #[cfg(not(target_arch = "wasm32"))]
    let root = std::env::temp_dir().join("runmat-explicit-mapped");
    let provider = Arc::new(CountedProvider {
        root,
        bytes: Arc::new(Mutex::new(vec![0; 8])),
        opens: Arc::new(AtomicUsize::new(0)),
        reads: Arc::new(AtomicUsize::new(0)),
        writes: Arc::new(AtomicUsize::new(0)),
        pending_reads: false,
        backing_drops: Arc::new(AtomicUsize::new(0)),
    });
    let _fs_selected = vfs::replace_provider(provider.clone());
    let mapped = futures::executor::block_on(super::super::super::super::memmapfile_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("data.bin".into()),
            Value::String("Format".into()),
            Value::String("double".into()),
            Value::String("Writable".into()),
            Value::Bool(true),
        ]),
    ))
    .unwrap();
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let rejected = futures::executor::block_on(subsasgn(
        None,
        mapped.clone(),
        property("Data"),
        Value::GpuTensor(explicit.clone()),
    ))
    .unwrap_err();
    assert_eq!(
        rejected.identifier(),
        MEMMAPFILE_EXPLICIT_GPU_EXTENSION.error_identifier
    );
    assert_eq!(gpu.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.reads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.writes.load(Ordering::SeqCst), 0);
    drop(strict);

    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let updated = futures::executor::block_on(subsasgn(
        None,
        mapped,
        property("Data"),
        Value::GpuTensor(explicit),
    ))
    .unwrap();
    assert!(matches!(updated.value(), Value::Object(_)));
    assert_eq!(gpu.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.writes.load(Ordering::SeqCst), 1);
    assert_eq!(*provider.bytes.lock().unwrap(), 7.0f64.to_le_bytes());
    assert_eq!(allocation.probe.retirements(), 0);
}
