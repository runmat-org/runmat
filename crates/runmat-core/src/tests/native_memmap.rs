//! Core evaluator ownership when a mapped byte read suspends.

use super::*;
use async_trait::async_trait;
use futures::channel::oneshot;
use futures::task::noop_waker_ref;
use runmat_filesystem::{
    replace_provider, DirEntry, FileHandle, FsMappedBacking, FsMappedWriteFailure,
    FsMappedWriteOutcome, FsMetadata, FsProvider, MemoryFsProvider, OpenFlags,
};
use runmat_value::Value;
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

const SOURCE: &str = r#"
function resetMappedCount()
  global mappedCount
  mappedCount = 0;
end
function count = mappedRead(path)
  global mappedCount
  mappedCount = mappedCount + 1;
  m = memmapfile(path);
  value = m.Data(2);
  assert(isequal(value, uint8(2)));
  count = mappedCount;
end
"#;

#[derive(Default)]
struct Evidence {
    opens: AtomicUsize,
    reads: AtomicUsize,
    completed_reads: AtomicUsize,
    retired_reads: AtomicUsize,
    retired_backings: AtomicUsize,
}

struct ReadFuture(Arc<Evidence>);
impl Drop for ReadFuture {
    fn drop(&mut self) {
        self.0.retired_reads.fetch_add(1, Ordering::SeqCst);
    }
}

struct GatedBacking {
    inner: Box<dyn FsMappedBacking>,
    gate: Arc<Mutex<Option<oneshot::Receiver<()>>>>,
    evidence: Arc<Evidence>,
}

impl Drop for GatedBacking {
    fn drop(&mut self) {
        self.evidence
            .retired_backings
            .fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait(?Send)]
impl FsMappedBacking for GatedBacking {
    fn opened_extent(&self) -> u64 {
        self.inner.opened_extent()
    }
    fn writable(&self) -> bool {
        self.inner.writable()
    }
    fn check_current(&self) -> io::Result<()> {
        self.inner.check_current()
    }
    async fn read_at(&mut self, offset: u64, bytes: &mut [u8]) -> io::Result<usize> {
        self.evidence.reads.fetch_add(1, Ordering::SeqCst);
        let _retirement = ReadFuture(self.evidence.clone());
        let receiver = self.gate.lock().unwrap().take();
        if let Some(receiver) = receiver {
            receiver
                .await
                .map_err(|_| io::Error::other("mapped read gate was abandoned"))?;
        }
        let count = self.inner.read_at(offset, bytes).await?;
        self.evidence.completed_reads.fetch_add(1, Ordering::SeqCst);
        Ok(count)
    }
    async fn write_at(
        &mut self,
        offset: u64,
        bytes: &[u8],
    ) -> Result<FsMappedWriteOutcome, FsMappedWriteFailure> {
        self.inner.write_at(offset, bytes).await
    }
}

struct GatedProvider {
    memory: MemoryFsProvider,
    gate: Arc<Mutex<Option<oneshot::Receiver<()>>>>,
    evidence: Arc<Evidence>,
}

#[async_trait(?Send)]
impl FsProvider for GatedProvider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/mapped-core"))
    }
    async fn open_mapped_backing(
        &self,
        path: &Path,
        writable: bool,
    ) -> io::Result<Option<Box<dyn FsMappedBacking>>> {
        self.evidence.opens.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .memory
            .open_mapped_backing(path, writable)
            .await?
            .map(|inner| {
                Box::new(GatedBacking {
                    inner,
                    gate: self.gate.clone(),
                    evidence: self.evidence.clone(),
                }) as Box<dyn FsMappedBacking>
            }))
    }
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.memory.open(path, flags)
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
    async fn set_readonly(&self, path: &Path, value: bool) -> io::Result<()> {
        self.memory.set_readonly(path, value).await
    }
}

fn invocation(function: &str, outputs: usize) -> ProcedureInvocation {
    ProcedureInvocation {
        target: ProcedureTarget::Function(function.into()),
        arguments: (outputs != 0)
            .then(|| Value::String("/mapped-core/data.bin".into()))
            .into_iter()
            .collect(),
        requested_outputs: outputs,
    }
}

#[test]
fn core_dropped_pending_mapped_read_retires_owner_and_fresh_call_reads_exact_bytes() {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir("/mapped-core");
    memory
        .write_project_path("/mapped-core/data.bin", &[1, 2, 3, 4])
        .unwrap();
    let (release, receiver) = oneshot::channel();
    let evidence = Arc::new(Evidence::default());
    let provider = Arc::new(GatedProvider {
        memory,
        gate: Arc::new(Mutex::new(Some(receiver))),
        evidence: evidence.clone(),
    });
    let _owner = replace_provider(provider);
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new("mapped-core@1", "mappedRead.m", SOURCE),
            None,
        ))
        .unwrap();
        let control = InvocationControl::default().force_generic_native();
        block_on(session.invoke_executable(&unit, invocation("resetMappedCount", 0), &control))
            .unwrap();
        let mut pending =
            Box::pin(session.invoke_executable(&unit, invocation("mappedRead", 1), &control));
        for _ in 0..8 {
            let mut task = Context::from_waker(noop_waker_ref());
            if let Poll::Ready(outcome) = pending.as_mut().poll(&mut task) {
                panic!("mapped call completed before the backing read suspended: {outcome:?}");
            }
            if evidence.reads.load(Ordering::SeqCst) == 1 {
                break;
            }
        }
        assert_eq!(evidence.reads.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.completed_reads.load(Ordering::SeqCst), 0);
        drop(pending);
        assert!(
            release.send(()).is_err(),
            "dropped call retained mapped read"
        );
        assert_eq!(evidence.retired_reads.load(Ordering::SeqCst), 1);

        let fresh =
            block_on(session.invoke_executable(&unit, invocation("mappedRead", 1), &control))
                .unwrap();
        assert_eq!(fresh.as_slice(), &[Value::Num(2.0)]);
        assert_eq!(evidence.opens.load(Ordering::SeqCst), 2);
        assert_eq!(evidence.reads.load(Ordering::SeqCst), 2);
        assert_eq!(evidence.completed_reads.load(Ordering::SeqCst), 1);
        drop(fresh);
        drop(session);
        assert_eq!(evidence.retired_backings.load(Ordering::SeqCst), 2);
    });
}
